"""Inspect real embeddings or insert a small fixture into existing collections."""

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path

from qdrant_client import models

from dejaview_ml.config import DIMENSIONS, Settings
from dejaview_ml.embeddings import Encoders
from dejaview_ml.media import LIMITS, MEDIA_TYPES
from dejaview_ml.storage import VectorStore, point_id, point_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modality", choices=list(DIMENSIONS))
    parser.add_argument("file", type=Path)
    parser.add_argument("--content-type")
    parser.add_argument("--upsert", action="store_true", help="write points to existing Qdrant")
    parser.add_argument("--movie-id", type=int)
    parser.add_argument("--offset-sec", type=int)
    parser.add_argument("--text-kind", choices=["query", "passage"], default="query")
    args = parser.parse_args()
    if args.upsert and args.movie_id is None:
        parser.error("--upsert requires --movie-id")
    if args.modality == "text" and args.offset_sec is not None:
        parser.error("text must not have --offset-sec")
    if args.offset_sec is not None and args.offset_sec < 0:
        parser.error("--offset-sec must be nonnegative")
    content_type = args.content_type or mimetypes.guess_type(args.file)[0]
    if args.content_type is None:
        content_type = {"audio/x-wav": "audio/wav", "audio/x-m4a": "audio/mp4"}.get(
            content_type,
            content_type,
        )
    if args.modality != "text" and content_type not in MEDIA_TYPES[args.modality]:
        parser.error("unsupported file type; use --content-type for extensionless files")
    with args.file.open("rb") as stream:
        data = stream.read(LIMITS[args.modality] + 1)
    if not data or len(data) > LIMITS[args.modality]:
        parser.error("empty file or file exceeds modality size limit")
    settings = Settings(**{f"{kind}_enabled": kind == args.modality for kind in DIMENSIONS})
    encoders = Encoders(settings)
    encoders.load(args.modality)
    if args.modality == "text":
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            parser.error("text file must be UTF-8")
        vectors = encoders.encode_text(text, passage=args.upsert or args.text_kind == "passage")
        offsets = [None]
    else:
        vectors = encoders.encode(args.modality, data, content_type)
        offsets = [(args.offset_sec or 0) + index * 10 for index in range(len(vectors))]

    async def upsert():
        store = VectorStore(settings)
        try:
            if not await store.client.collection_exists(args.modality):
                raise RuntimeError("collection missing: start the ML service first")
            info = await store.client.get_collection(args.modality)
            config = info.config.params.vectors
            if not isinstance(config, models.VectorParams) or (
                config.size != DIMENSIONS[args.modality]
                or config.distance != models.Distance.COSINE
            ):
                raise RuntimeError("incompatible collection schema")
            points = [
                models.PointStruct(
                    id=point_id(args.modality, args.movie_id, offset),
                    vector=vector,
                    payload=point_payload(args.modality, args.movie_id, offset),
                )
                for vector, offset in zip(vectors, offsets, strict=True)
            ]
            await store.client.upsert(args.modality, points, wait=True)
            print(json.dumps({"point_ids": [point.id for point in points]}))
        finally:
            await store.close()

    if args.upsert:
        asyncio.run(upsert())
    else:
        print(
            json.dumps(
                {
                    "modality": args.modality,
                    "dimension": DIMENSIONS[args.modality],
                    "offsets_sec": offsets,
                    "vectors": vectors,
                }
            )
        )


if __name__ == "__main__":
    main()
