"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS, GRAPH_TERM_SLOTS, RAG_NEIGHBOR_PAGES,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str, neighbors: int = RAG_NEIGHBOR_PAGES) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。

    neighborsが1以上なら、各ページの前後neighborsページの本文も付ける（宿題・実装）。
    前後ページは同じ資料の中だけから取り、同じページは1回だけ返す。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    columns = "page_id, document_id, lecture, document_title, section, title, page, content"
    with open_index(RAG_DB) as db:
        vector = sqlite_vec.serialize_float32(embed_text(query.strip(), "RETRIEVAL_QUERY"))
        rows = db.execute(f"""
            SELECT {columns}, vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (vector, RAG_TOP_K))
        hits = [dict(row) for row in rows]
        if neighbors <= 0:
            return hits

        # ここから追加: ヒットした順に「前ページ → ヒット → 後ページ」と並べる。
        hit_ids = {hit["page_id"] for hit in hits}
        results, seen = [], set()
        for hit in hits:
            rows = db.execute(f"""
                SELECT {columns}, vec_distance_cosine(embedding, ?) AS distance
                FROM chunks WHERE document_id = ? AND page BETWEEN ? AND ? ORDER BY page
            """, (vector, hit["document_id"], hit["page"] - neighbors, hit["page"] + neighbors))
            for row in rows:
                page = dict(row)
                if page["page_id"] in seen:  # 重複させない
                    continue
                seen.add(page["page_id"])
                # hit: ベクトル検索で選ばれたページ / neighbor: 前後として付けたページ
                page["source"] = "hit" if page["page_id"] in hit_ids else "neighbor"
                results.append(page)
        return results


def query_distances(query: str) -> dict:
    """質問文と、索引にある全ページとの距離を返す（宿題・発展）。"""
    with open_index(RAG_DB) as db:
        vector = sqlite_vec.serialize_float32(embed_text(query.strip(), "RETRIEVAL_QUERY"))
        rows = db.execute(
            "SELECT page_id, vec_distance_cosine(embedding, ?) AS distance FROM chunks", (vector,))
        return {row["page_id"]: row["distance"] for row in rows}


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query, neighbors=0) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    limit = max(0, GRAPH_MAX_RESULTS - len(results))

    # 発展：用語経由の候補に専用の枠を取り、
    # 入口と珍しい用語を多く共有するページを優先する。
    term_slots = min(GRAPH_TERM_SLOTS, limit)
    shared = defaultdict(set)
    for seed in seeds:
        for term in adjacent[seed["page_id"]]:
            if nodes[term]["kind"] != "term":
                continue
            for page_id in adjacent[term]:
                if page_id in pages and page_id not in seed_ids:
                    shared[page_id].add(term)

    # 第2版：用語で集めた候補を、質問に近い順に並べる。
    distances = query_distances(query) if term_slots and shared else {}

    neighbor_part = expanded[:limit - term_slots]
    taken = {candidate["page_id"] for candidate in neighbor_part}
    term_part = []
    for page_id in sorted(shared, key=lambda p: (distances.get(p, 2.0), p)):
        if len(term_part) >= term_slots:
            break
        if page_id in taken:
            continue
        terms = sorted(shared[page_id], key=lambda t: (len(adjacent[t]), t))
        root = next(s["page_id"] for s in seeds if terms[0] in adjacent[s["page_id"]])
        term_part.append({**pages[page_id], "hops": 2,
                          "path": [{"id": n, "title": nodes[n]["title"]}
                                   for n in (root, terms[0], page_id)],
                           "shared_terms": [nodes[t]["title"] for t in terms],
                           "distance": distances.get(page_id)})
        taken.add(page_id)
    rest = [c for c in expanded if c["page_id"] not in taken]
    expanded = (neighbor_part + term_part + rest)[:limit]
    return {"seeds": results, "expanded": expanded, "max_hops": GRAPH_HOPS}



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"種別: {'前後ページ' if result.get('source') == 'neighbor' else '検索ヒット'}")
        print(f"\n{result['content']}")