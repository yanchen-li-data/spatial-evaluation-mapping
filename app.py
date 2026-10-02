# -*- coding: utf-8 -*-
"""
菊名駅 歩行者利便性・コンビニ勢力圏
新規出店シミュレーション

構成
----
- Google Colab:
    重い既存分析を事前計算し runtime_cache.pkl を生成
- VS Code + Streamlit:
    runtime cacheを読み込み、地図・UIを表示
- 新規店舗候補:
    道路網ON時は新店舗1件分のみ
    Single-source Dijkstra を実行
- 動的更新:
    単純直線距離勢力圏
    利便性スコア勢力圏
    利便性HeatMap
"""

from pathlib import Path
from copy import deepcopy
import json
import pickle
import time
import tempfile

import requests

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium

from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import unary_union


# ============================================================
# 0. ファイル設定
#    ローカルVS Code / Streamlit Community Cloud 両対応
# ============================================================
BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# ローカルVS Codeで使用するruntime cache
# ------------------------------------------------------------
LOCAL_CACHE_PATH = (
    BASE_DIR
    / "data"
    / "菊名駅_2000m_runtime_cache_20260928.pkl"
)


# ------------------------------------------------------------
# Streamlit Community Cloudで一時保存するruntime cache
# ------------------------------------------------------------
CLOUD_CACHE_PATH = (
    Path(tempfile.gettempdir())
    / "kikuna_runtime_cache_20260928.pkl"
)


# ------------------------------------------------------------
# GitHub Release設定
#
# GitHub Repository / Releaseを作成した後、
# GITHUB_USERNAME と、必要に応じて GITHUB_REPOSITORY を
# 実際の値へ変更してください。
#
# GitHub Releaseへアップロードするasset名は、
# 下記の RELEASE_ASSET_NAME と完全に同じ名前にします。
# ------------------------------------------------------------
GITHUB_USERNAME = "yanchen-li-data"
GITHUB_REPOSITORY = "spatial-evaluation-mapping"
RELEASE_ASSET_NAME = "kikuna_runtime_cache_20260928.pkl"

CACHE_URL = (
    f"https://github.com/{GITHUB_USERNAME}/"
    f"{GITHUB_REPOSITORY}/releases/latest/download/"
    f"{RELEASE_ASSET_NAME}"
)


MANIFEST_PATH = (
    BASE_DIR
    / "菊名駅_2000m_runtime_cache_20260928_manifest.json"
)


def resolve_cache_path():
    """
    runtime cacheの利用場所を決定する。

    ローカルVS Code:
        data/菊名駅_2000m_runtime_cache_20260928.pkl
        が存在すれば、そのファイルをそのまま使用する。

    Streamlit Community Cloud:
        Repository内にruntime cacheが存在しない場合、
        GitHub Releaseからrequestsのストリーミング方式で取得し、
        OSの一時フォルダへ保存して使用する。

    これにより、同じapp.pyを
    ローカル環境とCommunity Cloudの両方で利用できる。
    """

    # --------------------------------------------------------
    # 1. ローカルPCにruntime cacheがある場合
    # --------------------------------------------------------
    if LOCAL_CACHE_PATH.exists():
        return LOCAL_CACHE_PATH

    # --------------------------------------------------------
    # 2. Cloud上ですでに正常なcacheを取得済みの場合
    # --------------------------------------------------------
    if (
        CLOUD_CACHE_PATH.exists()
        and CLOUD_CACHE_PATH.stat().st_size > 0
    ):
        return CLOUD_CACHE_PATH

    # 0 byteの不完全ファイルが残っている場合は削除
    if CLOUD_CACHE_PATH.exists():
        CLOUD_CACHE_PATH.unlink()

    # --------------------------------------------------------
    # 3. Cloud初回起動:
    #    GitHub Releaseからruntime cacheをストリーミング取得
    # --------------------------------------------------------
    partial_path = Path(
        str(CLOUD_CACHE_PATH) + ".part"
    )

    # 前回の途中ダウンロードが残っていれば削除
    if partial_path.exists():
        partial_path.unlink()

    try:
        with requests.get(
            CACHE_URL,
            stream=True,
            timeout=(30, 600),
            allow_redirects=True,
            headers={
                "User-Agent":
                    "Mozilla/5.0 Streamlit-Community-Cloud",
                "Accept":
                    "application/octet-stream",
            },
        ) as response:

            # 403 / 404 / 5xx 等をここで検出
            response.raise_for_status()

            with partial_path.open("wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

        # ----------------------------------------------------
        # ダウンロード結果チェック
        # ----------------------------------------------------
        if not partial_path.exists():
            raise RuntimeError(
                "runtime cacheファイルが作成されませんでした。"
            )

        downloaded_size = (
            partial_path.stat().st_size
        )

        if downloaded_size == 0:
            raise RuntimeError(
                "runtime cacheのダウンロード結果が0 byteです。"
            )

        # 現在のcacheは約287MB。
        # HTMLエラーページ等を誤って保存した場合も検出できるよう、
        # 100MB未満なら異常として停止する。
        if downloaded_size < 100 * 1024 * 1024:
            raise RuntimeError(
                "runtime cacheのダウンロードサイズが"
                "想定より小さすぎます。 "
                f"取得サイズ: "
                f"{downloaded_size / 1024 / 1024:.1f} MB"
            )

        # ダウンロード完了後のみ正式名へ変更。
        # 途中ファイルをpickle.loadしないようにする。
        partial_path.replace(
            CLOUD_CACHE_PATH
        )

        return CLOUD_CACHE_PATH

    except Exception as e:
        if partial_path.exists():
            partial_path.unlink()

        raise RuntimeError(
            "GitHub Releaseからruntime cacheを"
            "ダウンロードできませんでした。\n"
            f"URL: {CACHE_URL}\n"
            f"Error: {e}"
        ) from e


# ============================================================
# 1. 表示・モデル設定
# ============================================================
PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]

NEW_STORE_NAME = "★ 新規コンビニ候補"
NEW_STORE_COLOR = "#ff1493"

# Colab側の現行モデルと同じ重み。
# 新規店舗を「総合」HeatMapへ反映するときに使用する。
METRICS_CONFIG = [
    {
        "name": "駅",
        "weight": 60,
    },
    {
        "name": "コンビニ",
        "weight": 40,
    },
]


# ============================================================
# 2. runtime cache読込
# ============================================================
@st.cache_resource(show_spinner=False)
def load_runtime_cache(cache_path: str):
    cache = Path(cache_path)

    if not cache.exists():
        raise FileNotFoundError(
            f"runtime cacheが見つかりません: {cache}"
        )

    started = time.perf_counter()

    with cache.open("rb") as f:
        runtime = pickle.load(f)

    required_keys = {
        "layers",
        "G",
        "sim_results",
        "step1_voronoi",
        "existing_territory_geojson",
        "convenience_stores",
        "conv_lambda",
        "grid_context",
        "analysis_boundary",
        "_cache_meta",
    }

    missing = sorted(
        required_keys - set(runtime.keys())
    )

    if missing:
        raise RuntimeError(
            "runtime cacheに必要な項目がありません: "
            + ", ".join(missing)
        )

    runtime["_load_seconds"] = (
        time.perf_counter() - started
    )

    return runtime


@st.cache_data(show_spinner=False)
def load_manifest(path: str):
    manifest_path = Path(path)

    if not manifest_path.exists():
        return {}

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# runtime cacheを先にロードし、
# 中心座標・解析半径はcache metadataから取得する。
try:
    # ローカルではdata/配下のpklを使用し、
    # Community CloudではGitHub Releaseから取得する。
    CACHE_PATH = resolve_cache_path()

    runtime = load_runtime_cache(
        str(CACHE_PATH)
    )

except Exception as e:
    st.set_page_config(
        page_title="菊名駅 空間分析Webアプリ",
        layout="wide",
    )
    st.error("runtime cacheを読み込めませんでした。")
    st.exception(e)
    st.info(
        "ローカル環境ではdataフォルダ内のruntime cache、"
        "Community CloudではGitHub Release上の"
        "runtime cacheとCACHE_URL設定を確認してください。"
    )
    st.stop()

manifest = load_manifest(
    str(MANIFEST_PATH)
)

cache_meta = runtime.get(
    "_cache_meta",
    {},
)

CENTER_LAT = float(
    cache_meta.get(
        "center_lat",
        manifest.get(
            "analysis",
            {},
        ).get(
            "center_lat",
            35.5097,
        ),
    )
)

CENTER_LON = float(
    cache_meta.get(
        "center_lon",
        manifest.get(
            "analysis",
            {},
        ).get(
            "center_lon",
            139.6306,
        ),
    )
)

SEARCH_DIST = int(
    cache_meta.get(
        "search_dist",
        manifest.get(
            "analysis",
            {},
        ).get(
            "search_dist_m",
            2000,
        ),
    )
)



# ============================================================
# 3. development.ipynbで検証済みの空間分析ロジック
# ============================================================
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    scipy.spatial.Voronoi で生成された無限Voronoi領域を
    地図上で扱える有限ポリゴンへ変換する補助関数。
    """
    if vor.points.shape[1] != 2:
        raise ValueError("2次元データのみ対応しています")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # 各基準点に接続する ridge（Voronoi境界）を整理
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # 入力したコンビニ1店舗につき1つのVoronoi領域を復元
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        # すでに有限領域ならそのまま利用
        if vertices and all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        # 無限方向へ伸びている辺を、十分遠い有限地点まで延長
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1

            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, normal)) * normal

            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # 頂点をポリゴンの周囲順に並び替える
        vs = np.asarray([new_vertices[v] for v in new_region])
        polygon_center = vs.mean(axis=0)
        angles = np.arctan2(
            vs[:, 1] - polygon_center[1],
            vs[:, 0] - polygon_center[0]
        )
        new_region = np.asarray(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def _prepare_convenience_store_targets(target_gdf):
    """
    コンビニGeoDataFrameへ、Step2で使用する店舗ID・店舗名を付与する。
    同一座標の重複OSMオブジェクトは1店舗として扱う。
    """
    if target_gdf is None or target_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    stores = target_gdf.copy().reset_index(drop=True)

    if "name" in stores.columns:
        stores["_store_name"] = stores["name"].fillna("名称不明").astype(str)
    else:
        stores["_store_name"] = [f"コンビニ_{i}" for i in range(len(stores))]

    # 同一位置の重複を排除
    stores["_store_lon_key"] = stores.geometry.x.round(7)
    stores["_store_lat_key"] = stores.geometry.y.round(7)
    stores = (
        stores
        .drop_duplicates(subset=["_store_lon_key", "_store_lat_key"])
        .reset_index(drop=True)
    )

    # 重複排除後に安定した連番IDを付与
    stores["_store_id"] = np.arange(len(stores), dtype=int)

    return stores


def build_convenience_score_territory_geojson(
    territory_points_by_pid,
    center_lat,
    center_lon,
    search_dist=1000,
    step1_voronoi_gdf=None
):
    """
    Step2のコンビニ利便性スコア勢力圏をGeoJSON化する。

    ・道路網ON (N1):
      14,400点の勢力店舗判定結果を小セル化し、店舗ごとにdissolveする。
      最終外枠はヒートマップと同じ正方形で厳密にクリップする。

    ・道路網OFF (N0):
      Step1の数学的Voronoiポリゴンをそのまま再利用する。
      これによりStep1とStep2(N0)の境界形状を完全に揃え、
      グリッド由来のギザギザ境界を発生させない。
    """
    empty_fc = {
        "type": "FeatureCollection",
        "features": []
    }

    if not territory_points_by_pid:
        return {}

    palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"
    ]

    # ヒートマップと完全に同じ正方形の分析範囲
    radius_degree = search_dist / 111000.0

    analysis_boundary_4326 = box(
        center_lon - radius_degree,
        center_lat - radius_degree,
        center_lon + radius_degree,
        center_lat + radius_degree
    )

    result = {}

    # ========================================================
    # 1. N0（道路網OFF）
    #    Step1のVoronoiをそのままStep2でも使う
    # ========================================================
    n0_records = territory_points_by_pid.get(
        "S0_G0_N0",
        []
    )

    # Step2のN0計算済み点から、店舗名単位のスコア・距離集計だけ作る。
    # ポリゴン形状そのものはStep1から流用する。
    summary_by_name = {}

    for r in n0_records:
        name = str(r.get("store_name", "名称不明"))

        summary_by_name.setdefault(
            name,
            {
                "scores": [],
                "costs": []
            }
        )

        if r.get("score") is not None:
            summary_by_name[name]["scores"].append(
                float(r["score"])
            )

        if (
            r.get("cost") is not None
            and np.isfinite(float(r["cost"]))
        ):
            summary_by_name[name]["costs"].append(
                float(r["cost"])
            )

    if (
        step1_voronoi_gdf is not None
        and not step1_voronoi_gdf.empty
    ):
        n0_gdf = step1_voronoi_gdf.copy()

        # 念のため、Step1もここで同じ正方形に再クリップする。
        n0_gdf["geometry"] = n0_gdf.geometry.apply(
            lambda geom: geom.intersection(
                analysis_boundary_4326
            )
        )

        n0_gdf = n0_gdf[
            ~n0_gdf.geometry.is_empty
        ].copy()

        # Step1で既に色が付いていればそれをそのまま使い、
        # Step1とStep2(N0)の表示色も揃える。
        if "_color" not in n0_gdf.columns:
            n0_gdf["_color"] = [
                palette[int(store_id) % len(palette)]
                for store_id in n0_gdf["store_id"]
            ]

        mean_scores = []
        max_scores = []
        min_scores = []
        mean_costs = []

        for _, row in n0_gdf.iterrows():
            name = str(row["store_name"])
            summary = summary_by_name.get(
                name,
                {
                    "scores": [],
                    "costs": []
                }
            )

            scores = summary["scores"]
            costs = summary["costs"]

            mean_scores.append(
                float(np.mean(scores))
                if scores
                else None
            )
            max_scores.append(
                float(np.max(scores))
                if scores
                else None
            )
            min_scores.append(
                float(np.min(scores))
                if scores
                else None
            )
            mean_costs.append(
                float(np.mean(costs))
                if costs
                else None
            )

        n0_gdf["mean_score"] = mean_scores
        n0_gdf["max_score"] = max_scores
        n0_gdf["min_score"] = min_scores
        n0_gdf["mean_cost_m"] = mean_costs

        base_n0 = json.loads(
            n0_gdf.to_json()
        )

    else:
        # Step1が作れない特殊ケースだけ、空GeoJSONとする。
        base_n0 = empty_fc

    result["S0_G0_N0"] = base_n0
    result["S1_G0_N0"] = base_n0
    result["S0_G1_N0"] = base_n0
    result["S1_G1_N0"] = base_n0

    # ========================================================
    # 2. N1（道路網ON）
    #    道路・坂・信号を反映したグリッド勢力圏
    # ========================================================
    n1_pids = [
        "S0_G0_N1",
        "S1_G0_N1",
        "S0_G1_N1",
        "S1_G1_N1",
    ]

    for pid in n1_pids:
        records = territory_points_by_pid.get(
            pid,
            []
        )

        if not records:
            result[pid] = empty_fc
            continue

        valid_records = [
            r for r in records
            if r.get("store_id") is not None
        ]

        if not valid_records:
            result[pid] = empty_fc
            continue

        # グリッド間隔を推定
        unique_lats = np.array(
            sorted(
                set(
                    float(r["lat"])
                    for r in valid_records
                )
            )
        )

        unique_lons = np.array(
            sorted(
                set(
                    float(r["lon"])
                    for r in valid_records
                )
            )
        )

        if len(unique_lats) > 1:
            lat_step = float(
                np.median(
                    np.diff(unique_lats)
                )
            )
        else:
            lat_step = (
                search_dist / 111000.0
            ) / 30.0

        if len(unique_lons) > 1:
            lon_step = float(
                np.median(
                    np.diff(unique_lons)
                )
            )
        else:
            lon_step = (
                search_dist / 111000.0
            ) / 30.0

        half_lat = abs(lat_step) / 2.0
        half_lon = abs(lon_step) / 2.0

        by_store = {}

        for r in valid_records:
            by_store.setdefault(
                int(r["store_id"]),
                []
            ).append(r)

        feature_rows = []

        for store_id, store_records in by_store.items():
            # 各評価点を1セルとしてポリゴン化
            cells = [
                box(
                    float(r["lon"]) - half_lon,
                    float(r["lat"]) - half_lat,
                    float(r["lon"]) + half_lon,
                    float(r["lat"]) + half_lat
                )
                for r in store_records
            ]

            territory_geom = unary_union(
                cells
            )

            if territory_geom.is_empty:
                continue

            # ヒートマップと同じ正方形に厳密クリップ
            clipped = territory_geom.intersection(
                analysis_boundary_4326
            )

            if clipped.is_empty:
                continue

            if not clipped.is_valid:
                clipped = clipped.buffer(0)

            scores = [
                float(r["score"])
                for r in store_records
                if r.get("score") is not None
            ]

            costs = [
                float(r["cost"])
                for r in store_records
                if (
                    r.get("cost") is not None
                    and np.isfinite(
                        float(r["cost"])
                    )
                )
            ]

            store_name = str(
                store_records[0]["store_name"]
            )

            feature_rows.append({
                "store_id": int(store_id),
                "store_name": store_name,
                "mean_score": (
                    float(np.mean(scores))
                    if scores
                    else None
                ),
                "max_score": (
                    float(np.max(scores))
                    if scores
                    else None
                ),
                "min_score": (
                    float(np.min(scores))
                    if scores
                    else None
                ),
                "mean_cost_m": (
                    float(np.mean(costs))
                    if costs
                    else None
                ),
                "_color": palette[
                    store_id % len(palette)
                ],
                "geometry": clipped
            })

        if not feature_rows:
            result[pid] = empty_fc
            continue

        gdf = gpd.GeoDataFrame(
            feature_rows,
            crs="EPSG:4326"
        )

        result[pid] = json.loads(
            gdf.to_json()
        )

    return result


def coord_key(lat, lon):
    return (round(float(lat), 8), round(float(lon), 8))


def selected_pattern_id(network_on, slope_on, signal_on):
    if not network_on:
        return "S0_G0_N0"

    return (
        f"S{1 if slope_on else 0}"
        f"_G{1 if signal_on else 0}"
        "_N1"
    )


def square_boundary(center_lat, center_lon, search_dist):
    radius_degree = search_dist / 111000.0
    return box(
        center_lon - radius_degree,
        center_lat - radius_degree,
        center_lon + radius_degree,
        center_lat + radius_degree,
    )


def territory_fc_for_display(feature_collection):
    """
    GeoJSON tooltip表示用の文字列列を追加する。
    """
    fc = deepcopy(feature_collection)

    for feature in fc.get("features", []):
        p = feature.setdefault("properties", {})

        score = p.get("mean_score")
        cost = p.get("mean_cost_m")

        p["mean_score_display"] = (
            f"{float(score) * 100:.1f}点"
            if score is not None
            else "-"
        )

        p["mean_cost_display"] = (
            f"{float(cost):.0f}m相当"
            if cost is not None
            else "-"
        )

    return fc


def geometry_area_km2(geometry_mapping):
    if not geometry_mapping:
        return 0.0

    geom = shape(geometry_mapping)

    if geom.is_empty:
        return 0.0

    temp = gpd.GeoDataFrame(
        geometry=[geom],
        crs="EPSG:4326"
    )

    metric_crs = temp.estimate_utm_crs()

    if metric_crs is None:
        return 0.0

    return float(
        temp.to_crs(metric_crs).geometry.area.iloc[0]
        / 1_000_000.0
    )


def build_exact_voronoi_with_candidate(
    existing_stores,
    candidate_lat,
    candidate_lon,
    center_lat,
    center_lon,
    search_dist,
):
    if existing_stores is None:
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            None,
        )

    stores = existing_stores.copy()

    if stores.empty:
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            None,
        )

    # _prepare_convenience_store_targetsで既存店IDを振り直すため、
    # nameとgeometryを中心に一度整理する。
    stores = stores.copy()

    if "name" not in stores.columns:
        stores["name"] = stores[
            "_store_name"
        ]

    stores["_is_candidate"] = False

    candidate = gpd.GeoDataFrame(
        {
            "name": [NEW_STORE_NAME],
            "_is_candidate": [True],
        },
        geometry=[
            Point(
                float(candidate_lon),
                float(candidate_lat),
            )
        ],
        crs="EPSG:4326",
    )

    combined = gpd.GeoDataFrame(
        pd.concat(
            [stores, candidate],
            ignore_index=True,
        ),
        crs="EPSG:4326",
    )

    combined = _prepare_convenience_store_targets(
        combined
    )

    candidate_rows = combined[
        combined.get(
            "_is_candidate",
            False,
        ) == True
    ]

    if candidate_rows.empty:
        # 既存店舗と完全同一位置などで重複除去された場合
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            None,
        )

    candidate_id = int(
        candidate_rows.iloc[0]["_store_id"]
    )

    metric_crs = combined.estimate_utm_crs()

    if metric_crs is None:
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            candidate_id,
        )

    combined_m = combined.to_crs(
        metric_crs
    )

    coords = np.column_stack([
        combined_m.geometry.x.values,
        combined_m.geometry.y.values,
    ])

    # Voronoiには3つ以上の一意な位置が必要。
    if len(coords) < 3:
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            candidate_id,
        )

    vor = Voronoi(coords)

    regions, vertices = (
        voronoi_finite_polygons_2d(
            vor,
            radius=search_dist * 5,
        )
    )

    rows = []

    for i, region in enumerate(regions):
        if not region:
            continue

        polygon = Polygon(
            vertices[region]
        )

        if polygon.is_empty:
            continue

        rows.append({
            "store_id": int(
                combined_m.iloc[i][
                    "_store_id"
                ]
            ),
            "store_name": str(
                combined_m.iloc[i][
                    "_store_name"
                ]
            ),
            "_is_candidate": bool(
                combined_m.iloc[i].get(
                    "_is_candidate",
                    False,
                )
            ),
            "geometry": polygon,
        })

    if not rows:
        return (
            gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            candidate_id,
        )

    gdf = gpd.GeoDataFrame(
        rows,
        crs=metric_crs,
    ).to_crs("EPSG:4326")

    boundary = square_boundary(
        center_lat,
        center_lon,
        search_dist,
    )

    gdf["geometry"] = (
        gdf.geometry.apply(
            lambda geom: geom.intersection(
                boundary
            )
        )
    )

    gdf = gdf[
        ~gdf.geometry.is_empty
    ].copy()

    gdf["_color"] = [
        (
            NEW_STORE_COLOR
            if bool(row["_is_candidate"])
            else PALETTE[
                int(row["store_id"])
                % len(PALETTE)
            ]
        )
        for _, row in gdf.iterrows()
    ]

    return gdf, candidate_id


def calculate_candidate_result(
    runtime,
    candidate_lat,
    candidate_lon,
    pattern_id,
):
    """
    道路網OFF:
      新候補店までの直線距離と既存最寄り店距離を比較。
      表示形状は既存店+新店の数学的Voronoiを利用。

    道路網ON:
      新候補店をsourceとしてSingle-source Dijkstraを1回実行し、
      既存Step2で保存済みの「既存最良コスト」と比較する。
    """
    G = runtime["G"]
    sim_results = runtime["sim_results"]
    stores = runtime[
        "convenience_stores"
    ]
    conv_lambda = runtime["conv_lambda"]

    candidate_id = int(len(stores))

    territory_points = (
        sim_results
        .get(
            "_convenience_territory_points",
            {},
        )
        .get(pattern_id, [])
    )

    if not territory_points:
        return {
            "updated_records": [],
            "territory_geojson": {
                "type": "FeatureCollection",
                "features": [],
            },
            "step1_candidate_gdf": None,
            "candidate_id": candidate_id,
            "candidate_node": None,
            "candidate_area_km2": 0.0,
            "candidate_mean_score": None,
            "candidate_mean_cost": None,
            "captured_from": {},
            "winning_score_by_key": {},
            "candidate_wins_by_key": {},
        }

    updated_records = []
    captured_from = {}
    winning_score_by_key = {}
    candidate_wins_by_key = {}

    # --------------------------------------------------------
    # N0: 直線距離
    # --------------------------------------------------------
    if pattern_id.endswith("_N0"):
        for old in territory_points:
            lat = float(old["lat"])
            lon = float(old["lon"])

            candidate_cost = float(
                ox.distance.great_circle(
                    lat,
                    lon,
                    candidate_lat,
                    candidate_lon,
                )
            )

            old_cost_raw = old.get("cost")
            old_cost = (
                float(old_cost_raw)
                if old_cost_raw is not None
                else float("inf")
            )

            candidate_score = float(
                np.exp(
                    -conv_lambda
                    * candidate_cost
                )
            )

            wins = (
                candidate_cost
                < old_cost
            )

            if wins:
                rec = {
                    "lat": lat,
                    "lon": lon,
                    "score": candidate_score,
                    "cost": candidate_cost,
                    "store_id": candidate_id,
                    "store_name": NEW_STORE_NAME,
                }

                original_name = str(
                    old.get(
                        "store_name",
                        "名称不明",
                    )
                )

                captured_from[
                    original_name
                ] = (
                    captured_from.get(
                        original_name,
                        0,
                    )
                    + 1
                )
            else:
                rec = dict(old)

            updated_records.append(rec)

            key = coord_key(lat, lon)

            winning_score_by_key[key] = float(
                rec.get("score", 0.0)
            )

            candidate_wins_by_key[key] = wins

        exact_gdf, exact_candidate_id = (
            build_exact_voronoi_with_candidate(
                existing_stores=stores,
                candidate_lat=candidate_lat,
                candidate_lon=candidate_lon,
                center_lat=CENTER_LAT,
                center_lon=CENTER_LON,
                search_dist=SEARCH_DIST,
            )
        )

        if exact_candidate_id is not None:
            candidate_id = exact_candidate_id

        # グリッドから得た平均値を正確なVoronoi形状へ付与
        summary = {}

        for rec in updated_records:
            sid = int(rec["store_id"])

            summary.setdefault(
                sid,
                {
                    "scores": [],
                    "costs": [],
                },
            )

            if rec.get("score") is not None:
                summary[sid][
                    "scores"
                ].append(
                    float(rec["score"])
                )

            if rec.get("cost") is not None:
                summary[sid][
                    "costs"
                ].append(
                    float(rec["cost"])
                )

        if (
            exact_gdf is not None
            and not exact_gdf.empty
        ):
            exact_gdf = exact_gdf.copy()

            exact_gdf["mean_score"] = [
                (
                    float(
                        np.mean(
                            summary.get(
                                int(sid),
                                {},
                            ).get(
                                "scores",
                                [],
                            )
                        )
                    )
                    if summary.get(
                        int(sid),
                        {},
                    ).get(
                        "scores",
                        [],
                    )
                    else None
                )
                for sid in exact_gdf[
                    "store_id"
                ]
            ]

            exact_gdf[
                "mean_cost_m"
            ] = [
                (
                    float(
                        np.mean(
                            summary.get(
                                int(sid),
                                {},
                            ).get(
                                "costs",
                                [],
                            )
                        )
                    )
                    if summary.get(
                        int(sid),
                        {},
                    ).get(
                        "costs",
                        [],
                    )
                    else None
                )
                for sid in exact_gdf[
                    "store_id"
                ]
            ]

            territory_geojson = (
                {
                    "type": (
                        "FeatureCollection"
                    ),
                    "features": (
                        __import__(
                            "json"
                        ).loads(
                            exact_gdf.to_json()
                        )["features"]
                    ),
                }
            )
        else:
            territory_geojson = {
                "type": "FeatureCollection",
                "features": [],
            }

        step1_candidate_gdf = exact_gdf
        candidate_node = None

    # --------------------------------------------------------
    # N1: Python側Single-source Dijkstra
    # --------------------------------------------------------
    else:
        candidate_node = ox.nearest_nodes(
            G,
            X=float(candidate_lon),
            Y=float(candidate_lat),
        )

        weight_key = f"w_{pattern_id}"

        # ★ 動的化の中心:
        # 新規店舗1店だけについてPython/NetworkXで最短経路計算
        candidate_network_dist = (
            nx.single_source_dijkstra_path_length(
                G,
                source=candidate_node,
                weight=weight_key,
            )
        )

        for old in territory_points:
            lat = float(old["lat"])
            lon = float(old["lon"])

            key = coord_key(
                lat,
                lon,
            )

            context = runtime[
                "grid_context"
            ].get(key)

            if context is None:
                nearest_node = (
                    ox.nearest_nodes(
                        G,
                        X=lon,
                        Y=lat,
                    )
                )

                node_data = G.nodes[
                    nearest_node
                ]

                access_m = int(
                    ox.distance.great_circle(
                        lat,
                        lon,
                        node_data["y"],
                        node_data["x"],
                    )
                )
            else:
                nearest_node = context[
                    "nearest_node"
                ]
                access_m = context[
                    "access_m"
                ]

            network_cost = (
                candidate_network_dist.get(
                    nearest_node,
                    float("inf"),
                )
            )

            if np.isfinite(
                network_cost
            ):
                candidate_cost = float(
                    access_m
                    + network_cost
                )

                candidate_score = float(
                    np.exp(
                        -conv_lambda
                        * candidate_cost
                    )
                )
            else:
                candidate_cost = (
                    float("inf")
                )
                candidate_score = 0.0

            old_cost_raw = old.get(
                "cost"
            )

            old_cost = (
                float(old_cost_raw)
                if old_cost_raw is not None
                else float("inf")
            )

            wins = (
                candidate_cost
                < old_cost
            )

            if wins:
                rec = {
                    "lat": lat,
                    "lon": lon,
                    "score": candidate_score,
                    "cost": (
                        candidate_cost
                        if np.isfinite(
                            candidate_cost
                        )
                        else None
                    ),
                    "store_id": candidate_id,
                    "store_name": NEW_STORE_NAME,
                }

                original_name = str(
                    old.get(
                        "store_name",
                        "名称不明",
                    )
                )

                captured_from[
                    original_name
                ] = (
                    captured_from.get(
                        original_name,
                        0,
                    )
                    + 1
                )
            else:
                rec = dict(old)

            updated_records.append(rec)

            winning_score_by_key[key] = float(
                rec.get("score", 0.0)
            )

            candidate_wins_by_key[key] = wins

        temp_result = (
            build_convenience_score_territory_geojson(
                territory_points_by_pid={
                    pattern_id: (
                        updated_records
                    )
                },
                center_lat=CENTER_LAT,
                center_lon=CENTER_LON,
                search_dist=SEARCH_DIST,
                step1_voronoi_gdf=None,
            )
        )

        territory_geojson = (
            temp_result.get(
                pattern_id,
                {
                    "type": (
                        "FeatureCollection"
                    ),
                    "features": [],
                },
            )
        )

        # 新店舗だけ目立つ色へ
        for feature in territory_geojson.get(
            "features",
            [],
        ):
            props = feature.setdefault(
                "properties",
                {},
            )

            if (
                props.get(
                    "store_name"
                )
                == NEW_STORE_NAME
            ):
                props[
                    "_color"
                ] = NEW_STORE_COLOR

        step1_candidate_gdf, _ = (
            build_exact_voronoi_with_candidate(
                existing_stores=stores,
                candidate_lat=candidate_lat,
                candidate_lon=candidate_lon,
                center_lat=CENTER_LAT,
                center_lon=CENTER_LON,
                search_dist=SEARCH_DIST,
            )
        )

    # --------------------------------------------------------
    # 候補店の統計
    # --------------------------------------------------------
    candidate_feature = None

    for feature in territory_geojson.get(
        "features",
        [],
    ):
        if (
            feature.get(
                "properties",
                {},
            ).get(
                "store_name"
            )
            == NEW_STORE_NAME
        ):
            candidate_feature = feature
            break

    if candidate_feature is not None:
        p = candidate_feature.get(
            "properties",
            {},
        )

        candidate_area_km2 = (
            geometry_area_km2(
                candidate_feature.get(
                    "geometry"
                )
            )
        )

        candidate_mean_score = p.get(
            "mean_score"
        )

        candidate_mean_cost = p.get(
            "mean_cost_m"
        )
    else:
        candidate_area_km2 = 0.0
        candidate_mean_score = None
        candidate_mean_cost = None

        candidate_records = [
            r
            for r in updated_records
            if r.get(
                "store_name"
            ) == NEW_STORE_NAME
        ]

        if candidate_records:
            candidate_mean_score = float(
                np.mean([
                    float(r["score"])
                    for r
                    in candidate_records
                ])
            )

            finite_costs = [
                float(r["cost"])
                for r
                in candidate_records
                if (
                    r.get("cost")
                    is not None
                    and np.isfinite(
                        float(r["cost"])
                    )
                )
            ]

            candidate_mean_cost = (
                float(
                    np.mean(
                        finite_costs
                    )
                )
                if finite_costs
                else None
            )

    return {
        "updated_records": (
            updated_records
        ),
        "territory_geojson": (
            territory_geojson
        ),
        "step1_candidate_gdf": (
            step1_candidate_gdf
        ),
        "candidate_id": candidate_id,
        "candidate_node": candidate_node,
        "candidate_area_km2": (
            candidate_area_km2
        ),
        "candidate_mean_score": (
            candidate_mean_score
        ),
        "candidate_mean_cost": (
            candidate_mean_cost
        ),
        "captured_from": captured_from,
        "winning_score_by_key": (
            winning_score_by_key
        ),
        "candidate_wins_by_key": (
            candidate_wins_by_key
        ),
    }


def heat_data_with_candidate(
    runtime,
    facility_mode,
    pattern_id,
    candidate_result,
):
    sim = runtime["sim_results"]

    base = sim[
        facility_mode
    ][pattern_id]

    if not candidate_result:
        return base

    new_conv = candidate_result[
        "winning_score_by_key"
    ]

    if not new_conv:
        return base

    if facility_mode == "駅":
        return base

    old_conv_rows = sim[
        "コンビニ"
    ][pattern_id]

    old_conv = {
        coord_key(
            row[0],
            row[1],
        ): float(row[2])
        for row in old_conv_rows
    }

    conv_weight = next(
        m["weight"]
        for m in METRICS_CONFIG
        if m["name"] == "コンビニ"
    )

    total_weight = sum(
        m["weight"]
        for m in METRICS_CONFIG
    )

    result = []

    for lat, lon, base_score in base:
        key = coord_key(
            lat,
            lon,
        )

        new_conv_score = new_conv.get(
            key,
            old_conv.get(
                key,
                0.0,
            ),
        )

        if facility_mode == "コンビニ":
            score = new_conv_score
        else:
            old_conv_score = (
                old_conv.get(
                    key,
                    0.0,
                )
            )

            score = float(
                base_score
                + (
                    new_conv_score
                    - old_conv_score
                )
                * conv_weight
                / total_weight
            )

            score = max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            )

        result.append([
            lat,
            lon,
            score,
        ])

    return result



# ============================================================
# 5. 描画用ヘルパー
# ============================================================
def _representative_point(geom):
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Point":
        return geom

    return geom.representative_point()


def get_slope_gdf(runtime):
    """
    道路Graph → slope表示用GeoDataFrame。
    runtime objectへ一度だけ保持し、毎回graph_to_gdfsしない。
    """
    if "_slope_gdf_local" in runtime:
        return runtime["_slope_gdf_local"]

    G = runtime["G"]

    try:
        _, edges = ox.graph_to_gdfs(G)

        cols = [
            c
            for c in [
                "slope_pct",
                "geometry",
            ]
            if c in edges.columns
        ]

        slope_gdf = edges[cols].copy()

        if "slope_pct" not in slope_gdf.columns:
            slope_gdf["slope_pct"] = 0.0

        runtime["_slope_gdf_local"] = slope_gdf

        return slope_gdf

    except Exception:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326",
        )


def add_heatmap_legend(m):
    """
    旧HTMLにあった歩行者利便性スコア凡例と
    シミュレーション評価基準の注記を地図上へ表示する。
    """
    legend_html = """
    <div style="
        position: fixed;
        bottom: 28px;
        left: 48px;
        width: 465px;
        max-width: calc(100vw - 96px);
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        border: 1px solid #555;
        border-radius: 10px;
        padding: 12px 16px 14px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     'Yu Gothic UI', 'Meiryo', sans-serif;
        color: #243447;
        line-height: 1.45;
    ">
        <div style="
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 8px;
        ">
            歩行者利便性スコア
        </div>

        <div style="
            height: 16px;
            border-radius: 3px;
            background: linear-gradient(
                to right,
                #0000ff 0%,
                #00d9ff 25%,
                #00ff69 50%,
                #fff000 75%,
                #ff1e00 100%
            );
        "></div>

        <div style="
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-top: 3px;
            margin-bottom: 10px;
        ">
            <span>0.0 (不便)</span>
            <span>0.5</span>
            <span>1.0 (快適)</span>
        </div>

        <div style="
            border: 1px solid #e1a900;
            border-radius: 5px;
            padding: 8px 10px;
            background: rgba(255,252,235,0.96);
            font-size: 12px;
        ">
            <div style="
                font-weight: 700;
                margin-bottom: 4px;
            ">
                ⚠️ シミュレーション評価基準に関する注記:
            </div>

            <div>
                本ヒートマップの評価点は、エリア内の単純な
                「施設数」ではなく、各対象施設までのアクセス距離
                （歩行負荷）に基づき算出されています。
            </div>

            <div style="margin-top: 4px;">
                ・道路網の迂回時:
                坂道・信号による時間ロスを反映した
                「実質距離」で動的評価。
            </div>

            <div>
                ・迂回オフ時:
                地形ポテンシャルの基準値となる
                「直線距離」のみで静的評価。
            </div>
        </div>
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )


def add_display_layers(
    m,
    runtime,
    show_slope_layer,
    show_highways,
    show_buildings,
    show_water,
    show_rail,
    show_station,
    show_convenience,
    show_signals,
):
    """
    HeatMap・勢力圏の塗りより後に追加し、
    道路・建物・施設シンボルを前面に見せる。

    OFFのレイヤーはFoliumへ追加しないため、
    不要な大量GeoJSONをブラウザへ送らない。
    """
    layers = runtime["layers"]

    # --------------------------------------------------------
    # 道路傾斜・坂道
    # --------------------------------------------------------
    if show_slope_layer:
        slope_gdf = get_slope_gdf(
            runtime
        )

        if (
            slope_gdf is not None
            and not slope_gdf.empty
        ):
            def slope_style(feature):
                slope = float(
                    feature.get(
                        "properties",
                        {},
                    ).get(
                        "slope_pct",
                        0.0,
                    )
                    or 0.0
                )

                if slope < 2.0:
                    color, weight = "#74c476", 1.5
                elif slope < 4.0:
                    color, weight = "#41ab5d", 2.0
                elif slope < 6.0:
                    color, weight = "#238b45", 2.5
                elif slope < 8.0:
                    color, weight = "#006d2c", 3.5
                else:
                    color, weight = "#00441b", 5.0

                return {
                    "color": color,
                    "weight": weight,
                    "opacity": 0.78,
                }

            folium.GeoJson(
                slope_gdf,
                name="超低層：道路の傾斜・坂道",
                style_function=slope_style,
                interactive=False,
            ).add_to(m)

    # --------------------------------------------------------
    # 車道・大通り
    # --------------------------------------------------------
    highways = layers.get(
        "highways"
    )

    if (
        show_highways
        and highways is not None
        and not highways.empty
    ):
        folium.GeoJson(
            highways[["geometry"]],
            name="🚗 車道・大通り",
            style_function=lambda _: {
                "color": "#FF9900",
                "weight": 4,
                "opacity": 0.88,
            },
            interactive=False,
        ).add_to(m)

    # --------------------------------------------------------
    # 建物
    # --------------------------------------------------------
    buildings = layers.get(
        "buildings"
    )

    if (
        show_buildings
        and buildings is not None
        and not buildings.empty
    ):
        folium.GeoJson(
            buildings,
            name="🏢 建物・シルエット",
            style_function=lambda _: {
                "fillColor": "#7d8790",
                "color": "#34424d",
                "weight": 0.55,
                "fillOpacity": 0.22,
                "opacity": 0.75,
            },
            interactive=False,
        ).add_to(m)

    # --------------------------------------------------------
    # 水域・河川
    # --------------------------------------------------------
    water = layers.get(
        "water"
    )

    if (
        show_water
        and water is not None
        and not water.empty
    ):
        folium.GeoJson(
            water,
            name="💧 水域・河川",
            style_function=lambda _: {
                "fillColor": "#4ea5ff",
                "color": "#2477c8",
                "weight": 1.4,
                "fillOpacity": 0.38,
                "opacity": 0.90,
            },
            interactive=False,
        ).add_to(m)

    # --------------------------------------------------------
    # 鉄道
    # --------------------------------------------------------
    rail = layers.get(
        "rail"
    )

    if (
        show_rail
        and rail is not None
        and not rail.empty
    ):
        rail_group = folium.FeatureGroup(
            name="🚆 鉄道路線",
        )

        folium.GeoJson(
            rail,
            style_function=lambda _: {
                "color": "#222222",
                "weight": 3.5,
                "opacity": 0.82,
            },
            interactive=False,
        ).add_to(
            rail_group
        )

        folium.GeoJson(
            rail,
            style_function=lambda _: {
                "color": "#FFFFFF",
                "weight": 1.6,
                "dashArray": "6, 6",
                "opacity": 0.92,
            },
            interactive=False,
        ).add_to(
            rail_group
        )

        rail_group.add_to(m)

    # --------------------------------------------------------
    # 駅 / コンビニ
    # コンビニは赤から緑へ変更
    # --------------------------------------------------------
    amenities = layers.get(
        "amenities_gdf"
    )

    if (
        amenities is not None
        and not amenities.empty
        and "_mode" in amenities.columns
    ):
        facility_settings = [
            (
                "駅",
                show_station,
                "#1f77b4",
                "🚉 施設：駅",
            ),
            (
                "コンビニ",
                show_convenience,
                "#18a558",
                "🟢 施設：コンビニ",
            ),
        ]

        for (
            mode,
            show_mode,
            color,
            layer_name,
        ) in facility_settings:

            if not show_mode:
                continue

            sub = amenities[
                amenities["_mode"]
                .astype(str)
                == mode
            ]

            if sub.empty:
                continue

            group = folium.FeatureGroup(
                name=layer_name,
            )

            for _, row in sub.iterrows():
                p = _representative_point(
                    row.geometry
                )

                if p is None:
                    continue

                folium.CircleMarker(
                    location=[
                        p.y,
                        p.x,
                    ],
                    radius=5.5,
                    color="#ffffff",
                    weight=1.2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.96,
                    popup=str(
                        row.get(
                            "name",
                            mode,
                        )
                    ),
                ).add_to(
                    group
                )

            group.add_to(m)

    # --------------------------------------------------------
    # 信号機
    # --------------------------------------------------------
    signals = layers.get(
        "signal_gdf"
    )

    if (
        show_signals
        and signals is not None
        and not signals.empty
    ):
        group = folium.FeatureGroup(
            name="🚨 信号機（ペナルティ対象）",
        )

        for _, row in signals.iterrows():
            p = _representative_point(
                row.geometry
            )

            if p is None:
                continue

            folium.CircleMarker(
                location=[
                    p.y,
                    p.x,
                ],
                radius=3.5,
                color="#ffffff",
                weight=1,
                fill=True,
                fill_color="#e31a1c",
                fill_opacity=0.98,
                tooltip="信号機",
            ).add_to(
                group
            )

        group.add_to(m)


def add_straight_territory_fill(
    m,
    voronoi_gdf,
):
    """
    単純直線距離勢力圏の薄い塗り。
    """
    if (
        voronoi_gdf is None
        or voronoi_gdf.empty
    ):
        return

    folium.GeoJson(
        voronoi_gdf,
        name="🏪 単純直線距離勢力圏",
        style_function=lambda f: {
            "fillColor": (
                f["properties"].get(
                    "_color",
                    "#377eb8",
                )
            ),
            "color": (
                f["properties"].get(
                    "_color",
                    "#377eb8",
                )
            ),
            "weight": 0.7,
            "opacity": 0.38,
            "fillOpacity": 0.10,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["store_name"],
            aliases=["勢力店舗:"],
        ),
    ).add_to(m)


def add_straight_territory_border(
    m,
    voronoi_gdf,
):
    """
    単純直線距離勢力圏の強調境界。
    道路・建物より後に描画する。
    """
    if (
        voronoi_gdf is None
        or voronoi_gdf.empty
    ):
        return

    folium.GeoJson(
        voronoi_gdf,
        name="🔲 単純直線距離勢力圏の強調境界",
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "#202020",
            "weight": 2.4,
            "opacity": 0.92,
            "dashArray": "7, 5",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["store_name"],
            aliases=["勢力店舗:"],
        ),
    ).add_to(m)


def add_convenience_territory_fill(
    m,
    feature_collection,
):
    """
    利便性スコア勢力圏の薄い塗り。
    """
    if not feature_collection:
        return

    fc = territory_fc_for_display(
        feature_collection
    )

    def style_function(feature):
        p = feature.get(
            "properties",
            {},
        )

        color = p.get(
            "_color",
            "#8e44ad",
        )

        return {
            "fillColor": color,
            "fillOpacity": 0.15,
            "color": color,
            "weight": 0.7,
            "opacity": 0.40,
        }

    folium.GeoJson(
        fc,
        name="🏪 利便性スコア勢力圏",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "store_name",
                "mean_score_display",
                "mean_cost_display",
            ],
            aliases=[
                "勢力店舗:",
                "勢力圏内平均スコア:",
                "平均実質歩行負荷:",
            ],
            localize=True,
        ),
    ).add_to(m)


def add_convenience_territory_border(
    m,
    feature_collection,
):
    """
    利便性スコア勢力圏の強調境界。
    道路・建物より後に描画する。
    """
    if not feature_collection:
        return

    fc = territory_fc_for_display(
        feature_collection
    )

    folium.GeoJson(
        fc,
        name="🔲 利便性スコア勢力圏の強調境界",
        style_function=lambda _: {
            "fillOpacity": 0,
            "color": "#5b2c83",
            "weight": 2.6,
            "opacity": 0.96,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "store_name",
                "mean_score_display",
                "mean_cost_display",
            ],
            aliases=[
                "勢力店舗:",
                "勢力圏内平均スコア:",
                "平均実質歩行負荷:",
            ],
            localize=True,
        ),
    ).add_to(m)


def get_evaluation_popup_data(
    runtime,
    evaluation_click,
):
    """
    クリック位置に最も近い事前計算済み評価グリッドを探し、
    地図上ポップアップ表示用の情報を返す。

    ポップアップの表示位置は「実際にクリックした地点」、
    内容は従来HTMLと同じ _click_popups のHTMLを利用する。
    """
    if not evaluation_click:
        return None

    click_popups = (
        runtime["sim_results"].get(
            "_click_popups",
            [],
        )
    )

    if not click_popups:
        return None

    lat0 = float(
        evaluation_click["lat"]
    )
    lon0 = float(
        evaluation_click["lon"]
    )

    best = min(
        click_popups,
        key=lambda row: (
            (
                float(row[0])
                - lat0
            ) ** 2
            + (
                float(row[1])
                - lon0
            ) ** 2
        ),
    )

    return {
        "lat": lat0,
        "lon": lon0,
        "html": best[2],
        "grid_lat": float(best[0]),
        "grid_lon": float(best[1]),
    }


def add_evaluation_popup(
    m,
    evaluation_popup,
):
    """
    地点的利便性評価をクリック地点に自動表示する。

    透明なCircleMarkerをアンカーにし、
    元HTMLと同様にLeaflet Popupとして地図上へ表示する。
    """
    if not evaluation_popup:
        return

    popup = folium.Popup(
        evaluation_popup["html"],
        max_width=480,
        min_width=300,
        show=True,
    )

    folium.CircleMarker(
        location=[
            evaluation_popup["lat"],
            evaluation_popup["lon"],
        ],
        radius=1,
        color="transparent",
        weight=0,
        opacity=0.0,
        fill=True,
        fill_color="transparent",
        fill_opacity=0.0,
        popup=popup,
    ).add_to(m)


def build_streamlit_map(
    runtime,
    facility_mode,
    pattern_id,
    show_slope_layer,
    show_highways,
    show_buildings,
    show_water,
    show_rail,
    show_station,
    show_convenience,
    show_signals,
    show_heatmap,
    heatmap_strength,
    show_straight_territory,
    show_straight_border,
    show_convenience_territory,
    show_convenience_border,
    candidate_lat,
    candidate_lon,
    candidate_result,
    reflect_candidate_in_heatmap,
    evaluation_popup=None,
):
    m = folium.Map(
        location=[
            CENTER_LAT,
            CENTER_LON,
        ],
        zoom_start=16,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    # ========================================================
    # 描画順
    # 1. HeatMap（最背面）
    # 2. 勢力圏の塗り
    # 3. 道路・建物・水域・鉄道・施設
    # 4. 勢力圏の強調境界
    # 5. 新規候補マーカー
    # ========================================================

    # --------------------------------------------------------
    # 1. HeatMap
    # --------------------------------------------------------
    if show_heatmap:
        if (
            candidate_result
            and reflect_candidate_in_heatmap
        ):
            heat_data = (
                heat_data_with_candidate(
                    runtime=runtime,
                    facility_mode=facility_mode,
                    pattern_id=pattern_id,
                    candidate_result=(
                        candidate_result
                    ),
                )
            )
        else:
            heat_data = (
                runtime["sim_results"]
                .get(
                    facility_mode,
                    {},
                )
                .get(
                    pattern_id,
                    [],
                )
            )

        processed = [
            [
                float(r[0]),
                float(r[1]),
                max(
                    0.005,
                    min(
                        1.0,
                        float(r[2])
                        * float(
                            heatmap_strength
                        ),
                    ),
                ),
            ]
            for r in heat_data
        ]

        if processed:
            plugins.HeatMap(
                processed,
                name="💡 歩行者利便性スコア",
                radius=24,
                blur=18,
                min_opacity=0.06,
                max_zoom=18,
            ).add_to(m)

            add_heatmap_legend(m)

    # --------------------------------------------------------
    # 使用する単純直線距離勢力圏
    # 新規候補があれば、候補店を含む動的Voronoi。
    # --------------------------------------------------------
    if (
        candidate_result
        and candidate_result.get(
            "step1_candidate_gdf"
        ) is not None
        and not candidate_result[
            "step1_candidate_gdf"
        ].empty
    ):
        straight_territory_gdf = (
            candidate_result[
                "step1_candidate_gdf"
            ]
        )
    else:
        straight_territory_gdf = (
            runtime[
                "step1_voronoi"
            ]
        )

    # --------------------------------------------------------
    # 使用する利便性スコア勢力圏
    # --------------------------------------------------------
    if candidate_result:
        convenience_territory_fc = (
            candidate_result[
                "territory_geojson"
            ]
        )
    else:
        convenience_territory_fc = (
            runtime[
                "existing_territory_geojson"
            ].get(
                pattern_id,
                {
                    "type": "FeatureCollection",
                    "features": [],
                },
            )
        )

    # --------------------------------------------------------
    # 2. 勢力圏の塗り
    # --------------------------------------------------------
    if show_straight_territory:
        add_straight_territory_fill(
            m=m,
            voronoi_gdf=(
                straight_territory_gdf
            ),
        )

    if show_convenience_territory:
        add_convenience_territory_fill(
            m=m,
            feature_collection=(
                convenience_territory_fc
            ),
        )

    # --------------------------------------------------------
    # 3. 地図表示レイヤー
    # HeatMapより後に描画し、視認性を確保
    # --------------------------------------------------------
    add_display_layers(
        m=m,
        runtime=runtime,
        show_slope_layer=show_slope_layer,
        show_highways=show_highways,
        show_buildings=show_buildings,
        show_water=show_water,
        show_rail=show_rail,
        show_station=show_station,
        show_convenience=show_convenience,
        show_signals=show_signals,
    )

    # --------------------------------------------------------
    # 4. 勢力圏の強調境界
    # --------------------------------------------------------
    if (
        show_straight_territory
        and show_straight_border
    ):
        add_straight_territory_border(
            m=m,
            voronoi_gdf=(
                straight_territory_gdf
            ),
        )

    if (
        show_convenience_territory
        and show_convenience_border
    ):
        add_convenience_territory_border(
            m=m,
            feature_collection=(
                convenience_territory_fc
            ),
        )

    # --------------------------------------------------------
    # 5. 新規候補マーカー
    # --------------------------------------------------------
    if (
        candidate_lat is not None
        and candidate_lon is not None
    ):
        folium.Marker(
            location=[
                candidate_lat,
                candidate_lon,
            ],
            tooltip=NEW_STORE_NAME,
            popup=(
                f"{NEW_STORE_NAME}<br>"
                f"緯度: {candidate_lat:.6f}<br>"
                f"経度: {candidate_lon:.6f}"
            ),
            icon=folium.DivIcon(
                icon_size=(34, 34),
                icon_anchor=(17, 17),
                html=(
                    '<div style="'
                    'font-size:30px;'
                    'color:#ff1493;'
                    'text-shadow:'
                    '0 0 3px white,'
                    '0 0 3px white;'
                    '">★</div>'
                ),
            ),
        ).add_to(m)

    # --------------------------------------------------------
    # 6. 地点的利便性評価ポップアップ
    # --------------------------------------------------------
    add_evaluation_popup(
        m=m,
        evaluation_popup=evaluation_popup,
    )

    # Streamlitサイドバーが主操作だが、
    # 現在地図に追加されているレイヤーを確認できるよう
    # LayerControlは折りたたみ状態で残す。
    folium.LayerControl(
        collapsed=True,
    ).add_to(m)

    return m


# ============================================================
# 6. Streamlit UI
# ============================================================
st.set_page_config(
    page_title=(
        "歩行者利便性スコア・"
        "コンビニ勢力圏シミュレーション"
    ),
    layout="wide",
)

st.title(
    "歩行者利便性スコア・"
    "コンビニ勢力圏シミュレーション"
)

st.caption(
    "地図上で歩行者利便性とコンビニ勢力圏を確認し、"
    "新規コンビニ候補を指定すると、"
    "単純直線距離勢力圏・利便性スコア勢力圏・"
    "利便性HeatMapを動的に更新します。"
)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
st.session_state.setdefault(
    "candidate_lat",
    None,
)

st.session_state.setdefault(
    "candidate_lon",
    None,
)

st.session_state.setdefault(
    "candidate_cache_key",
    None,
)

st.session_state.setdefault(
    "candidate_cache_result",
    None,
)

st.session_state.setdefault(
    "last_handled_click",
    None,
)

st.session_state.setdefault(
    "evaluation_click",
    None,
)

st.session_state.setdefault(
    "straight_territory_border",
    False,
)

st.session_state.setdefault(
    "convenience_territory_border",
    True,
)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:

    # ========================================================
    # 1. 地図表示レイヤー
    # ========================================================
    st.subheader(
        "地図表示レイヤー"
    )

    show_slope_layer = st.checkbox(
        "超低層：道路の傾斜・坂道",
        value=False,
    )

    show_highways = st.checkbox(
        "🚗 車道・大通り",
        value=False,
    )

    show_buildings = st.checkbox(
        "🏢 建物・シルエット",
        value=False,
    )

    show_water = st.checkbox(
        "💧 水域・河川",
        value=False,
    )

    show_rail = st.checkbox(
        "🚆 鉄道路線",
        value=True,
    )

    show_station = st.checkbox(
        "🚉 施設：駅",
        value=True,
    )

    show_convenience = st.checkbox(
        "🟢 施設：コンビニ",
        value=True,
    )

    show_signals = st.checkbox(
        "🚨 信号機（ペナルティ対象）",
        value=False,
    )

    st.divider()

    # ========================================================
    # 2. 利便性スコアシミュレーション
    # ========================================================
    st.subheader(
        "利便性スコアシミュレーション"
    )

    show_heatmap = st.checkbox(
        "💡 歩行者利便性スコアを表示",
        value=True,
    )

    facility_label = st.radio(
        "🎯 対象施設の選択",
        [
            "総合",
            "駅のみ",
            "コンビニのみ",
        ],
        index=0,
        horizontal=True,
    )

    facility_mode = {
        "総合": "ALL",
        "駅のみ": "駅",
        "コンビニのみ": "コンビニ",
    }[facility_label]

    network_on = st.checkbox(
        "道路網の迂回（経路探索を有効化）",
        value=True,
    )

    slope_on = st.checkbox(
        "┗ 🚨 坂道（高低差の歩行負荷を算入）",
        value=True,
        disabled=not network_on,
    )

    signal_on = st.checkbox(
        "┗ 🚥 信号機（交差点タイムロスを算入）",
        value=True,
        disabled=not network_on,
    )

    if not network_on:
        slope_on = False
        signal_on = False

    pattern_id = selected_pattern_id(
        network_on=network_on,
        slope_on=slope_on,
        signal_on=signal_on,
    )

    condition_text = (
        f"現在の評価条件："
        f"道路網 {'ON' if network_on else 'OFF'} ｜ "
        f"坂道 {'ON' if slope_on else 'OFF'} ｜ "
        f"信号 {'ON' if signal_on else 'OFF'}"
    )

    st.caption(
        condition_text
    )

    if show_heatmap:
        heatmap_strength = st.slider(
            "HeatMapの濃さ",
            min_value=0.35,
            max_value=1.00,
            value=1.00,
            step=0.05,
            help=(
                "値を小さくすると、道路・建物・勢力圏などを"
                "HeatMapの上から確認しやすくなります。"
            ),
        )
    else:
        heatmap_strength = 1.00

    st.divider()

    # ========================================================
    # 3. コンビニ勢力圏
    # ========================================================
    st.subheader(
        "コンビニ勢力圏"
    )

    show_straight_territory = st.checkbox(
        "🏪 単純直線距離勢力圏",
        value=False,
    )

    if not show_straight_territory:
        st.session_state[
            "straight_territory_border"
        ] = False

    show_straight_border = st.checkbox(
        "┗ 🔲 勢力圏の境界線を強調",
        key="straight_territory_border",
        disabled=not show_straight_territory,
    )

    st.caption(
        "道路網・坂道・信号機の設定には連動せず、"
        "各コンビニまでの直線距離だけで勢力圏を判定します。"
    )

    show_convenience_territory = st.checkbox(
        "🏪 利便性スコア勢力圏",
        value=True,
    )

    if not show_convenience_territory:
        st.session_state[
            "convenience_territory_border"
        ] = False

    show_convenience_border = st.checkbox(
        "┗ 🔲 勢力圏の境界線を強調",
        key="convenience_territory_border",
        disabled=not show_convenience_territory,
    )

    st.caption(
        "道路網・坂道・信号機の現在のチェック条件に連動します。\n"
        "道路網OFF時は直線距離による最寄りコンビニになります。"
    )

    st.divider()

    # ========================================================
    # 4. 新規コンビニ出店シミュレーション
    # ========================================================
    st.subheader(
        "🏪 新規コンビニ出店シミュレーション"
    )

    operation_mode = st.radio(
        "地図クリック時の動作",
        [
            "新規店舗候補を指定",
            "地点的利便性を確認",
        ],
        index=0,
    )

    st.caption(
        "「新規店舗候補を指定」の状態で"
        "地図上の任意地点をクリックすると、"
        "新規コンビニ候補として動的計算します。"
    )

    reflect_candidate_in_heatmap = (
        st.checkbox(
            "新店舗を利便性HeatMapにも反映",
            value=True,
            disabled=not show_heatmap,
        )
    )

    candidate_lat = (
        st.session_state[
            "candidate_lat"
        ]
    )

    candidate_lon = (
        st.session_state[
            "candidate_lon"
        ]
    )

    if (
        candidate_lat is not None
        and candidate_lon is not None
    ):
        st.success(
            "現在の候補地点\n\n"
            f"緯度: {candidate_lat:.6f}\n\n"
            f"経度: {candidate_lon:.6f}"
        )

    with st.expander(
        "緯度・経度を手入力"
    ):
        manual_lat = st.number_input(
            "緯度",
            value=float(
                candidate_lat
                if candidate_lat is not None
                else CENTER_LAT
            ),
            format="%.7f",
        )

        manual_lon = st.number_input(
            "経度",
            value=float(
                candidate_lon
                if candidate_lon is not None
                else CENTER_LON
            ),
            format="%.7f",
        )

        if st.button(
            "この座標を候補地点に設定",
            use_container_width=True,
        ):
            st.session_state[
                "candidate_lat"
            ] = float(
                manual_lat
            )

            st.session_state[
                "candidate_lon"
            ] = float(
                manual_lon
            )

            st.session_state[
                "candidate_cache_key"
            ] = None

            st.session_state[
                "candidate_cache_result"
            ] = None

            st.session_state[
                "evaluation_click"
            ] = None

            st.rerun()

    if st.button(
        "候補地点をクリア",
        use_container_width=True,
    ):
        st.session_state[
            "candidate_lat"
        ] = None

        st.session_state[
            "candidate_lon"
        ] = None

        st.session_state[
            "candidate_cache_key"
        ] = None

        st.session_state[
            "candidate_cache_result"
        ] = None

        st.session_state[
            "evaluation_click"
        ] = None

        st.rerun()


# ============================================================
# 7. 新規候補の動的計算
# ============================================================
candidate_lat = st.session_state[
    "candidate_lat"
]

candidate_lon = st.session_state[
    "candidate_lon"
]

candidate_result = None
candidate_error = None

if (
    candidate_lat is not None
    and candidate_lon is not None
):
    radius_degree = (
        SEARCH_DIST
        / 111000.0
    )

    inside = (
        abs(
            candidate_lat
            - CENTER_LAT
        )
        <= radius_degree
        and abs(
            candidate_lon
            - CENTER_LON
        )
        <= radius_degree
    )

    if not inside:
        st.warning(
            "候補地点が現在の分析正方形範囲外です。"
            "道路Graphや事前計算データの範囲不足により、"
            "結果が不完全になる可能性があります。"
        )

    cache_key = (
        round(
            float(candidate_lat),
            7,
        ),
        round(
            float(candidate_lon),
            7,
        ),
        pattern_id,
    )

    if (
        st.session_state[
            "candidate_cache_key"
        ]
        == cache_key
        and st.session_state[
            "candidate_cache_result"
        ]
        is not None
    ):
        candidate_result = (
            st.session_state[
                "candidate_cache_result"
            ]
        )

    else:
        try:
            with st.spinner(
                (
                    "新規店舗を含む単純直線距離勢力圏を計算中..."
                    if not network_on
                    else (
                        "新規店舗1件分の経路探索を行い、"
                        "利便性スコア勢力圏を更新中..."
                    )
                )
            ):
                started = (
                    time.perf_counter()
                )

                candidate_result = (
                    calculate_candidate_result(
                        runtime=runtime,
                        candidate_lat=(
                            candidate_lat
                        ),
                        candidate_lon=(
                            candidate_lon
                        ),
                        pattern_id=(
                            pattern_id
                        ),
                    )
                )

                candidate_result[
                    "_calculation_seconds"
                ] = (
                    time.perf_counter()
                    - started
                )

            st.session_state[
                "candidate_cache_key"
            ] = cache_key

            st.session_state[
                "candidate_cache_result"
            ] = candidate_result

        except ImportError as e:
            candidate_error = str(e)

            st.error(
                "新規店舗計算に必要なライブラリが不足しています。"
            )

            st.code(
                "python -m pip install scikit-learn",
                language="powershell",
            )

            st.exception(e)

        except Exception as e:
            candidate_error = str(e)

            st.error(
                "新規店舗の動的計算中にエラーが発生しました。"
            )

            st.exception(e)


# ============================================================
# 8. 候補店分析結果
# ============================================================
if candidate_result:

    st.subheader(
        "新規候補店の分析結果"
    )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "利便性スコア勢力圏面積",
        (
            f"{candidate_result['candidate_area_km2']:.3f} km²"
        ),
    )

    mean_score = candidate_result[
        "candidate_mean_score"
    ]

    c2.metric(
        "勢力圏内平均スコア",
        (
            f"{mean_score * 100:.1f}点"
            if mean_score is not None
            else "-"
        ),
    )

    mean_cost = candidate_result[
        "candidate_mean_cost"
    ]

    c3.metric(
        "平均実質歩行負荷",
        (
            f"{mean_cost:.0f}m相当"
            if mean_cost is not None
            else "-"
        ),
    )

    c4.metric(
        "計算方式",
        (
            "道路ネットワーク経路探索"
            if network_on
            else "直線距離"
        ),
    )

    c5.metric(
        "動的計算時間",
        (
            f"{candidate_result.get('_calculation_seconds', 0):.2f}秒"
        ),
    )

    captured = candidate_result.get(
        "captured_from",
        {},
    )

    winning_grids = sum(
        1
        for is_win
        in candidate_result.get(
            "candidate_wins_by_key",
            {},
        ).values()
        if is_win
    )

    st.caption(
        f"新規店舗が獲得した評価グリッド: "
        f"{winning_grids:,} 点"
    )

    if captured:
        captured_df = pd.DataFrame(
            [
                {
                    "既存勢力店舗": name,
                    "新店舗が獲得した評価グリッド数": count,
                }
                for name, count
                in sorted(
                    captured.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ]
        )

        with st.expander(
            "どの既存店舗から勢力圏を獲得したか"
        ):
            st.dataframe(
                captured_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 9. 地図生成
# ============================================================
evaluation_popup = None

if (
    operation_mode
    == "地点的利便性を確認"
):
    evaluation_popup = (
        get_evaluation_popup_data(
            runtime=runtime,
            evaluation_click=(
                st.session_state.get(
                    "evaluation_click"
                )
            ),
        )
    )

map_obj = build_streamlit_map(
    runtime=runtime,
    facility_mode=facility_mode,
    pattern_id=pattern_id,
    show_slope_layer=show_slope_layer,
    show_highways=show_highways,
    show_buildings=show_buildings,
    show_water=show_water,
    show_rail=show_rail,
    show_station=show_station,
    show_convenience=show_convenience,
    show_signals=show_signals,
    show_heatmap=show_heatmap,
    heatmap_strength=heatmap_strength,
    show_straight_territory=(
        show_straight_territory
    ),
    show_straight_border=(
        show_straight_border
    ),
    show_convenience_territory=(
        show_convenience_territory
    ),
    show_convenience_border=(
        show_convenience_border
    ),
    candidate_lat=candidate_lat,
    candidate_lon=candidate_lon,
    candidate_result=candidate_result,
    reflect_candidate_in_heatmap=(
        reflect_candidate_in_heatmap
    ),
    evaluation_popup=evaluation_popup,
)

st.subheader(
    "地図"
)

st.caption(
    condition_text
)

if operation_mode == "新規店舗候補を指定":
    st.caption(
        "地図をクリックすると、その位置を"
        "新規コンビニ候補として設定します。"
    )
else:
    st.caption(
        "地図をクリックすると、クリックした場所に"
        "地点的利便性評価のポップアップを表示します。"
    )

map_output = st_folium(
    map_obj,
    height=760,
    use_container_width=True,
    key="main_map",
    returned_objects=[
        "last_clicked",
    ],
)

clicked = (
    map_output.get(
        "last_clicked"
    )
    if map_output
    else None
)

if clicked:
    lat_clicked = float(
        clicked["lat"]
    )

    lon_clicked = float(
        clicked["lng"]
    )

    handled_key = (
        operation_mode,
        round(
            lat_clicked,
            7,
        ),
        round(
            lon_clicked,
            7,
        ),
    )

    if (
        handled_key
        != st.session_state[
            "last_handled_click"
        ]
    ):
        st.session_state[
            "last_handled_click"
        ] = handled_key

        if (
            operation_mode
            == "新規店舗候補を指定"
        ):
            st.session_state[
                "candidate_lat"
            ] = lat_clicked

            st.session_state[
                "candidate_lon"
            ] = lon_clicked

            st.session_state[
                "candidate_cache_key"
            ] = None

            st.session_state[
                "candidate_cache_result"
            ] = None

            st.session_state[
                "evaluation_click"
            ] = None

            st.rerun()

        else:
            st.session_state[
                "evaluation_click"
            ] = {
                "lat": lat_clicked,
                "lon": lon_clicked,
            }

            # 地図はすでに描画済みなので、
            # 再描画してクリック地点へPopupを追加する。
            st.rerun()


# ============================================================
# 10. 地点的利便性確認
# ============================================================
# 地点評価は地図下部へ出力せず、
# 地図上のクリック地点にLeaflet Popupとして表示する。


# ============================================================
# 11. 補足
# ============================================================
with st.expander(
    "このアプリの計算方法"
):
    st.markdown(
        """
**単純直線距離勢力圏**

既存コンビニと新規候補店の座標から
数学的Voronoiを計算します。
道路・坂道・信号機は考慮せず、
純粋な直線距離だけで最寄り店舗を判定します。

**利便性スコア勢力圏**

道路網ON時は、道路ネットワーク上の迂回に加え、
現在ONになっている坂道・信号機の歩行負荷を反映します。
新規候補店については1店舗分だけ
Single-source Dijkstraを実行し、
既存店舗についてはColabで事前計算済みの
最良コストを再利用します。

道路網OFF時は直線距離による最寄りコンビニとなるため、
境界形状は単純直線距離勢力圏と一致します。

**利便性HeatMap**

「新店舗を利便性HeatMapにも反映」がONの場合、
新規候補によって改善したコンビニ利便性を
HeatMapへ反映します。

「総合」ではコンビニ成分だけを更新し、
「駅のみ」は新規コンビニの影響を受けません。
        """
    )
