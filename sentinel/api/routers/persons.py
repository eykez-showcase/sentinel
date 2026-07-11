from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.core.database import get_db
from sentinel.models.person import FaceEmbedding, Person
from sentinel.schemas.person import PersonCreate, PersonRead, PersonUpdate, FaceEmbeddingRead

router = APIRouter()

PHOTOS_DIR = Path("config/persons")
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def _get_recognizer():
    """Get the shared FaceRecognizer from app state — injected at startup."""
    from sentinel.api.main import face_recognizer
    return face_recognizer


@router.get("", response_model=list[PersonRead])
async def list_persons(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Person).order_by(Person.created_at.desc()))
    persons = result.scalars().all()

    out = []
    for p in persons:
        count_result = await db.execute(
            select(func.count()).where(FaceEmbedding.person_id == p.id)
        )
        photo_count = count_result.scalar() or 0
        pr = PersonRead.model_validate(p)
        pr.photo_count = photo_count
        out.append(pr)
    return out


@router.post("", response_model=PersonRead, status_code=201)
async def create_person(body: PersonCreate, db: AsyncSession = Depends(get_db)):
    person = Person(name=body.name, role=body.role, notes=body.notes)
    db.add(person)
    await db.commit()
    await db.refresh(person)
    pr = PersonRead.model_validate(person)
    pr.photo_count = 0
    return pr


@router.get("/{person_id}", response_model=PersonRead)
async def get_person(person_id: str, db: AsyncSession = Depends(get_db)):
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person not found")
    count_result = await db.execute(
        select(func.count()).where(FaceEmbedding.person_id == person_id)
    )
    pr = PersonRead.model_validate(person)
    pr.photo_count = count_result.scalar() or 0
    return pr


@router.patch("/{person_id}", response_model=PersonRead)
async def update_person(person_id: str, body: PersonUpdate, db: AsyncSession = Depends(get_db)):
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(person, field, value)
    await db.commit()
    await db.refresh(person)
    count_result = await db.execute(
        select(func.count()).where(FaceEmbedding.person_id == person_id)
    )
    pr = PersonRead.model_validate(person)
    pr.photo_count = count_result.scalar() or 0
    return pr


@router.delete("/{person_id}", status_code=204)
async def delete_person(person_id: str, db: AsyncSession = Depends(get_db)):
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person not found")
    await db.delete(person)
    await db.commit()
    # Reload recognizer embeddings after deletion
    await _reload_recognizer(db)


@router.post("/{person_id}/photos", response_model=FaceEmbeddingRead, status_code=201)
async def upload_photo(
    person_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    person = await db.get(Person, person_id)
    if not person:
        raise HTTPException(404, "Person not found")

    recognizer = _get_recognizer()
    if not recognizer.available:
        raise HTTPException(503, "Face recognition not available — install insightface on the Jetson")

    import cv2
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not decode image")

    embedding = recognizer.extract_embedding(image)
    if embedding is None:
        raise HTTPException(422, "No face detected in this photo — try a clearer front-facing photo")

    # Save photo to disk
    photo_dir = PHOTOS_DIR / person_id
    photo_dir.mkdir(parents=True, exist_ok=True)
    photo_path = str(photo_dir / f"{uuid.uuid4()}.jpg")
    cv2.imwrite(photo_path, image)

    fe = FaceEmbedding(
        person_id=person_id,
        embedding=embedding.tobytes(),
        photo_path=photo_path,
    )
    db.add(fe)
    await db.commit()
    await db.refresh(fe)

    # Reload recognizer with updated embeddings
    await _reload_recognizer(db)

    return FaceEmbeddingRead.model_validate(fe)


@router.get("/{person_id}/photos", response_model=list[FaceEmbeddingRead])
async def list_photos(person_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(FaceEmbedding)
        .where(FaceEmbedding.person_id == person_id)
        .order_by(FaceEmbedding.created_at)
    )
    return [FaceEmbeddingRead.model_validate(fe) for fe in result.scalars().all()]


@router.delete("/{person_id}/photos/{photo_id}", status_code=204)
async def delete_photo(person_id: str, photo_id: str, db: AsyncSession = Depends(get_db)):
    fe = await db.get(FaceEmbedding, photo_id)
    if not fe or fe.person_id != person_id:
        raise HTTPException(404, "Photo not found")
    if fe.photo_path:
        Path(fe.photo_path).unlink(missing_ok=True)
    await db.delete(fe)
    await db.commit()
    await _reload_recognizer(db)


async def _reload_recognizer(db: AsyncSession) -> None:
    """Reload all embeddings into the in-memory recognizer."""
    from sentinel.api.main import face_recognizer
    result = await db.execute(
        select(FaceEmbedding.person_id, Person.name, Person.role, FaceEmbedding.embedding)
        .join(Person, Person.id == FaceEmbedding.person_id)
    )
    rows = result.all()
    face_recognizer.load_embeddings([(r[0], r[1], r[2], r[3]) for r in rows])
