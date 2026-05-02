from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.db import get_db
from pipeline.models import Template
from pipeline.schemas import TemplateCreate, TemplateResponse

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("")
async def list_templates(db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(select(Template).order_by(Template.is_builtin.desc(), Template.name))
    templates = result.scalars().all()
    return {"templates": [TemplateResponse.model_validate(t) for t in templates]}


@router.get("/{name}")
async def get_template(name: str, db: AsyncSession = Depends(get_db)) -> dict:
    result = await db.execute(select(Template).where(Template.name == name))
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"template": TemplateResponse.model_validate(template)}


@router.post("", status_code=201)
async def create_template(body: TemplateCreate, db: AsyncSession = Depends(get_db)) -> dict:
    existing = await db.execute(select(Template).where(Template.name == body.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Template name already exists")
    template = Template(name=body.name, label=body.label, description=body.description, chain=body.chain, is_builtin=False)
    db.add(template)
    await db.commit()
    await db.refresh(template)
    return {"template": TemplateResponse.model_validate(template)}


@router.delete("/{name}", status_code=204)
async def delete_template(name: str, db: AsyncSession = Depends(get_db)) -> None:
    result = await db.execute(select(Template).where(Template.name == name))
    template = result.scalar_one_or_none()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    if template.is_builtin:
        raise HTTPException(status_code=403, detail="Cannot delete a built-in template")
    await db.delete(template)
    await db.commit()
