import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")


class LLMExplainer:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 api_base: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_base = (api_base or os.getenv("OPENROUTER_BASE_URL")
                         or "https://openrouter.ai/api/v1")

    def _call_llm(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error calling LLM: {e}]"

    def _build_prompt(self, rec: dict, game_details: dict) -> str:
        title = game_details.get("title", rec["game_id"])
        year = game_details.get("year", "unknown")
        genre = game_details.get("genre", "unknown")
        platforms = ", ".join(game_details.get("platforms", [])) or "unknown"
        score = rec.get("final_score", 0)
        content = rec.get("content_score", 0)
        pop = rec.get("popularity_score", 0)
        conf = rec.get("data_confidence", 50)

        return f"""You are a game recommendation expert. Explain in 1-2 sentences why this game was recommended.

Game: {title} ({year})
Genre: {genre}
Platforms: {platforms}
Content similarity: {content:.2f}
Popularity: {pop:.2f}
Data confidence: {conf}/100
Final score: {score:.2f}

Write a natural, concise explanation in Spanish. Focus on what makes this game a good fit."""

    def explain(self, rec: dict) -> str:
        game_id = rec["game_id"]
        details = self._get_game_details(game_id)
        prompt = self._build_prompt(rec, details)
        return self._call_llm(prompt)

    def explain_batch(self, recommendations: list[dict]) -> list[dict]:
        out = []
        for rec in recommendations:
            explanation = self.explain(rec)
            out.append({**rec, "llm_explanation": explanation})
        return out

    @staticmethod
    def _get_game_details(game_id: str) -> dict:
        games = pl.read_parquet(RAW_DIR / "games.parquet")
        row = games.filter(pl.col("game_id") == game_id)
        if len(row) == 0:
            return {}
        r = row.to_dicts()[0]
        platforms = pl.read_parquet(RAW_DIR / "game_platforms.parquet")
        platform_ids = platforms.filter(pl.col("game_id") == game_id)["platform_id"].to_list()
        plat_cat = pl.read_parquet(RAW_DIR / "platforms.parquet")
        plat_names = plat_cat.filter(pl.col("id").is_in(platform_ids))["name"].to_list()
        return {
            "title": r.get("title", ""),
            "year": r.get("release_year", ""),
            "genre": r.get("genre_id", ""),
            "platforms": plat_names,
        }


def compare_explanations(rule_based: str, llm_based: str) -> dict:
    return {
        "rule_based_chars": len(rule_based),
        "llm_based_chars": len(llm_based),
        "llm_is_longer": len(llm_based) > len(rule_based),
    }
