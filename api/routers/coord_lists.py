"""Shared, named coordinate lists: preview, save, list, rename, delete.

Anyone can see, apply, rename or delete any saved list; no owner is recorded.
Deleting only hides a list, so it can be restored in the database.
"""
from __future__ import annotations

import os
import sqlite3

from fastapi import APIRouter, Body, Depends, HTTPException

from api import coord_lists
from api.db import get_coord_lists, get_science

router = APIRouter()


def _checked(text: str, radius_arcsec: float) -> None:
    if len(text) > coord_lists.MAX_TEXT_CHARS:
        raise HTTPException(413, f'file over {coord_lists.MAX_TEXT_CHARS:,} characters')
    if not 0 < radius_arcsec <= coord_lists.MAX_RADIUS_ARCSEC:
        raise HTTPException(422, f'radius must be in (0, {coord_lists.MAX_RADIUS_ARCSEC}] arcsec')
    n = sum(1 for line in text.splitlines() if line.split('#', 1)[0].strip())
    if n > coord_lists.MAX_COORDS:
        raise HTTPException(413, f'{n:,} lines; at most {coord_lists.MAX_COORDS:,} positions')


def _name(name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(422, 'the list needs a name')
    if len(name) > coord_lists.MAX_NAME_CHARS:
        raise HTTPException(422, f'name over {coord_lists.MAX_NAME_CHARS} characters')
    return name


def _stamp(sci: sqlite3.Connection):
    """Changes when the science database file does, so cached positions are rebuilt."""
    path = sci.execute('PRAGMA database_list').fetchone()['file']
    return (path, os.path.getmtime(path)) if path else ('memory', id(sci))


@router.post('/coord-lists/preview')
def preview_coord_list(text: str = Body(..., description="one 'ra dec' per line, decimal degrees or sexagesimal"),
                       radius_arcsec: float = Body(coord_lists.DEFAULT_RADIUS_ARCSEC),
                       sci: sqlite3.Connection = Depends(get_science)):
    """What a list would match, without saving it."""
    _checked(text, radius_arcsec)
    return coord_lists.evaluate(sci, _stamp(sci), text, radius_arcsec)[0]


@router.post('/coord-lists')
def create_coord_list(text: str = Body(..., description="one 'ra dec' per line, decimal degrees or sexagesimal"),
                      name: str = Body(...),
                      radius_arcsec: float = Body(coord_lists.DEFAULT_RADIUS_ARCSEC),
                      sci: sqlite3.Connection = Depends(get_science),
                      store: sqlite3.Connection = Depends(get_coord_lists)):
    """Match and save a list for everyone; pass its `id` as `coord_list=` to the list endpoints."""
    _checked(text, radius_arcsec)
    return coord_lists.create(store, sci, _stamp(sci), text, _name(name), radius_arcsec)


@router.get('/coord-lists')
def list_coord_lists(store: sqlite3.Connection = Depends(get_coord_lists)):
    """Every saved list, newest first."""
    return coord_lists.all_lists(store)


@router.get('/coord-lists/{list_id}')
def get_coord_list(list_id: str, store: sqlite3.Connection = Depends(get_coord_lists)):
    found = coord_lists.get(store, list_id)
    if found is None:
        raise HTTPException(404, f'no coordinate list {list_id}')
    return found


def _existing(action):
    try:
        return action()
    except KeyError:
        raise HTTPException(404, 'no such coordinate list') from None


@router.patch('/coord-lists/{list_id}')
def rename_coord_list(list_id: str, name: str = Body(..., embed=True),
                      store: sqlite3.Connection = Depends(get_coord_lists)):
    return _existing(lambda: coord_lists.rename(store, list_id, _name(name)))


@router.delete('/coord-lists/{list_id}')
def delete_coord_list(list_id: str, store: sqlite3.Connection = Depends(get_coord_lists)):
    """Hide the list from everyone; links to it stop filtering. Recoverable in the database."""
    _existing(lambda: coord_lists.delete(store, list_id))
    return {'deleted': list_id}


def coord_list_filter(store: sqlite3.Connection, list_id: str | None):
    """`coord_list=` as a filter clause, for the list endpoints; 404 for an unknown id."""
    try:
        return coord_lists.as_filter(store, list_id)
    except KeyError:
        raise HTTPException(404, f'no coordinate list {list_id}; it may have been deleted') from None
