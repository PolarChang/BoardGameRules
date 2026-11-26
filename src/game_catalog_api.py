"""Game selection endpoints shared by local and full knowledge services."""
from fastapi import APIRouter, Header, HTTPException
from src.database import get_all_games, get_game, create_game
from src.editorial_api import admin
from src.models import Game

router = APIRouter(tags=["knowledge"])


@router.get("/games")
async def api_list_games():
    """列出所有遊戲。"""
    games = get_all_games()
    return {
        "games": [
            {
                "id": g.id,
                "name": g.name,
                "publisher": g.publisher,
                "designer": g.designer,
                "player_count": g.player_count,
                "play_time": g.play_time,
                "complexity": g.complexity,
                "status": g.status.value,
            }
            for g in games
        ]
    }


@router.post("/games")
async def api_create_game(game: Game, x_knowledge_admin: str = Header(default="")):
    """建立遊戲（需管理員授權）。"""
    admin(x_knowledge_admin)
    existing = get_game(game.id)
    if existing:
        raise HTTPException(status_code=400, detail=f"遊戲已存在: {game.id}")
    created = create_game(game)
    return created.model_dump()


@router.get("/games/{game_id}")
async def api_get_game(game_id: str):
    """取得遊戲詳細資訊。"""
    game = get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")
    return game.model_dump()


