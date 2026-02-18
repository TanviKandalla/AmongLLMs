"""OpenSkill rating service for updating model ratings after games.

Uses a "Meta-Agent" approach to handle asymmetric team sizes (2 impostors vs 5 crewmates).
Instead of rating teams directly (which causes 27x asymmetry in rating changes due to
OpenSkill's probability model treating larger teams as favored), we collapse each team
to a single meta-agent with averaged mu/sigma, run a 1v1 match, and distribute the
symmetric deltas to all team members.

The delta is distributed using variance-weighted redistribution: players with higher
uncertainty (sigma) receive larger updates. This is the correct Bayesian approach -
uncertain players have more "room to grow" while stable players are already well-known.
With equal sigmas, everyone gets the same delta (backward compatible).

Note: OpenSkill's `weights` parameter was tested but found to only affect intra-team
credit distribution, not inter-team balance. The meta-agent approach is the correct
solution for symmetric team-level deltas.
"""

import math

from openskill.models import PlackettLuce
from sqlalchemy.orm import Session

from app.models import Model, ModelRating, Game, GameParticipant, PlayerRole


# Configure OpenSkill model
# PlackettLuce is a good default for team-based games
RATING_MODEL = PlackettLuce()


def get_or_create_rating(db: Session, model: Model) -> ModelRating:
    """Get existing rating or create a new one for a model."""
    if model.ratings is not None:
        return model.ratings

    rating = ModelRating(model_id=model.id)
    db.add(rating)
    db.flush()
    model.ratings = rating
    return rating


def update_ratings_for_game(db: Session, game: Game) -> None:
    """
    Update OpenSkill ratings for all participants after a game completes.

    Uses a "Meta-Agent" approach to handle asymmetric team sizes:
    1. Collapse each team to a single meta-agent (average mu, sqrt of avg variance for sigma)
    2. Run a 1v1 match between meta-agents
    3. Distribute deltas using variance-weighted redistribution

    Variance weighting means players with higher uncertainty (sigma) receive
    larger updates. This is Bayesian-correct: uncertain players converge faster,
    stable players change less. With equal sigmas, everyone gets the same delta.
    """
    if game.winner is None:
        return

    # Separate participants by role
    impostors: list[GameParticipant] = []
    crewmates: list[GameParticipant] = []

    for participant in game.participants:
        if participant.role == PlayerRole.IMPOSTOR:
            impostors.append(participant)
        else:
            crewmates.append(participant)

    # Determine which team won
    impostors_won = game.impostors_won

    # Build OpenSkill rating objects for each team
    impostor_ratings = []
    crewmate_ratings = []

    for p in impostors:
        model_rating = get_or_create_rating(db, p.model)
        os_rating = RATING_MODEL.rating(
            mu=model_rating.impostor_mu, sigma=model_rating.impostor_sigma
        )
        impostor_ratings.append((p, model_rating, os_rating))

    for p in crewmates:
        model_rating = get_or_create_rating(db, p.model)
        os_rating = RATING_MODEL.rating(
            mu=model_rating.crewmate_mu, sigma=model_rating.crewmate_sigma
        )
        crewmate_ratings.append((p, model_rating, os_rating))

    # Create meta-agents by averaging team mu and using sqrt(avg variance) for sigma
    impostor_team = [r[2] for r in impostor_ratings]
    crewmate_team = [r[2] for r in crewmate_ratings]

    def create_meta_agent(team: list) -> object:
        """Create a meta-agent with averaged mu and sqrt of averaged variance."""
        avg_mu = sum(r.mu for r in team) / len(team)
        avg_variance = sum(r.sigma**2 for r in team) / len(team)
        return RATING_MODEL.rating(mu=avg_mu, sigma=math.sqrt(avg_variance))

    meta_impostor = create_meta_agent(impostor_team)
    meta_crewmate = create_meta_agent(crewmate_team)

    # Run 1v1 match between meta-agents
    # OpenSkill ranks: lower is better, so winner gets rank 0
    if impostors_won:
        ranks = [0, 1]  # Impostors won (rank 0), crewmates lost (rank 1)
    else:
        ranks = [1, 0]  # Impostors lost (rank 1), crewmates won (rank 0)

    new_meta_imp, new_meta_crew = RATING_MODEL.rate(
        [[meta_impostor], [meta_crewmate]],
        ranks=ranks,
    )

    # Calculate team-level deltas and sigma ratios
    impostor_mu_delta = new_meta_imp[0].mu - meta_impostor.mu
    crewmate_mu_delta = new_meta_crew[0].mu - meta_crewmate.mu
    impostor_sigma_ratio = new_meta_imp[0].sigma / meta_impostor.sigma
    crewmate_sigma_ratio = new_meta_crew[0].sigma / meta_crewmate.sigma

    # Distribute deltas using variance-weighted redistribution
    # Players with higher sigma (more uncertainty) get larger updates
    impostor_total_variance = sum(r.sigma**2 for r in impostor_team)
    impostor_pool = impostor_mu_delta * len(impostor_team)

    for participant, model_rating, old_rating in impostor_ratings:
        share = old_rating.sigma**2 / impostor_total_variance
        model_rating.impostor_mu = old_rating.mu + (impostor_pool * share)
        model_rating.impostor_sigma = max(0.1, old_rating.sigma * impostor_sigma_ratio)
        model_rating.impostor_games += 1
        if impostors_won:
            model_rating.impostor_wins += 1

    crewmate_total_variance = sum(r.sigma**2 for r in crewmate_team)
    crewmate_pool = crewmate_mu_delta * len(crewmate_team)

    for participant, model_rating, old_rating in crewmate_ratings:
        share = old_rating.sigma**2 / crewmate_total_variance
        model_rating.crewmate_mu = old_rating.mu + (crewmate_pool * share)
        model_rating.crewmate_sigma = max(0.1, old_rating.sigma * crewmate_sigma_ratio)
        model_rating.crewmate_games += 1
        if not impostors_won:
            model_rating.crewmate_wins += 1

    db.flush()


def scale_rating_for_display(mu: float) -> int:
    """
    Scale OpenSkill mu to a display-friendly integer.

    OpenSkill default mu is 25. We multiply by 100 for bigger numbers.
    25 * 100 = 2500 starting rating.
    """
    return round(mu * 100)


def _build_rankings_from_ratings(
    models: list[Model], ratings_map: dict[str, ModelRating]
) -> list[dict]:
    """
    Build ranked leaderboard dicts from models and a ratings map.

    Args:
        models: All registered models.
        ratings_map: Map of model.id -> ModelRating (real or temporary).

    Returns:
        List of dicts ready for the leaderboard API, sorted by conservative rating.
    """
    rankings = []
    for model in models:
        rating = ratings_map.get(model.id, ModelRating(model_id=model.id))

        rankings.append(
            {
                "model": model,
                "rating": rating,
                "overall": rating.overall_rating,
                "overall_sigma": rating.overall_sigma,
                "conservative": rating.conservative_rating,
                "impostor": rating.impostor_rating,
                "impostor_sigma": rating.impostor_sigma or ModelRating.DEFAULT_SIGMA,
                "crewmate": rating.crewmate_rating,
                "crewmate_sigma": rating.crewmate_sigma or ModelRating.DEFAULT_SIGMA,
                "games": rating.total_games,
            }
        )

    rankings.sort(key=lambda x: x["conservative"], reverse=True)

    result = []
    for i, r in enumerate(rankings, start=1):
        model = r["model"]
        rating = r["rating"]

        result.append(
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "provider": model.provider,
                "overall_rating": scale_rating_for_display(r["overall"]),
                "impostor_rating": scale_rating_for_display(r["impostor"]),
                "crewmate_rating": scale_rating_for_display(r["crewmate"]),
                "overall_sigma": scale_rating_for_display(r["overall_sigma"]),
                "impostor_sigma": scale_rating_for_display(r["impostor_sigma"]),
                "crewmate_sigma": scale_rating_for_display(r["crewmate_sigma"]),
                "games_played": r["games"],
                "current_rank": i,
                "impostor_games": rating.impostor_games,
                "impostor_wins": rating.impostor_wins,
                "crewmate_games": rating.crewmate_games,
                "crewmate_wins": rating.crewmate_wins,
                "win_rate": round(rating.overall_win_rate * 100, 1),
                "impostor_win_rate": round(rating.impostor_win_rate * 100, 1),
                "crewmate_win_rate": round(rating.crewmate_win_rate * 100, 1),
                "release_date": model.release_date.isoformat() if model.release_date else None,
                "avatar_color": model.avatar_color,
            }
        )

    return result


def get_model_rankings(db: Session) -> list[dict]:
    """
    Get all models with their current persisted ratings, ranked by overall rating.

    Returns a list of dicts ready for the leaderboard API response.
    """
    models = db.query(Model).all()
    ratings_map = {}
    for model in models:
        if model.ratings is not None:
            ratings_map[model.id] = model.ratings
    return _build_rankings_from_ratings(models, ratings_map)


def get_historical_rankings(db: Session, engine_version: int) -> list[dict]:
    """
    Compute rankings for a past season by replaying its games in memory.

    Creates temporary ModelRating objects (NOT added to the DB session),
    replays all completed games for the given engine_version chronologically,
    and returns rankings sorted by conservative_rating.
    """
    from app.models import Game, GameStatus

    models = db.query(Model).all()

    # Create in-memory ratings (not attached to session)
    temp_ratings: dict[str, ModelRating] = {}
    for model in models:
        r = ModelRating(
            impostor_mu=ModelRating.DEFAULT_MU,
            impostor_sigma=ModelRating.DEFAULT_SIGMA,
            impostor_games=0,
            impostor_wins=0,
            crewmate_mu=ModelRating.DEFAULT_MU,
            crewmate_sigma=ModelRating.DEFAULT_SIGMA,
            crewmate_games=0,
            crewmate_wins=0,
        )
        temp_ratings[model.id] = r

    # Get completed games for this engine version
    games = (
        db.query(Game)
        .filter(Game.status == GameStatus.COMPLETED, Game.engine_version == engine_version)
        .order_by(Game.ended_at.asc())
        .all()
    )

    # Replay each game using the same meta-agent logic
    for game in games:
        if game.winner is None:
            continue

        impostors_won = game.impostors_won

        impostor_data = []
        crewmate_data = []

        for p in game.participants:
            r = temp_ratings.get(p.model_id)
            if r is None:
                continue
            if p.role == PlayerRole.IMPOSTOR:
                os_r = RATING_MODEL.rating(mu=r.impostor_mu, sigma=r.impostor_sigma)
                impostor_data.append((p, r, os_r))
            else:
                os_r = RATING_MODEL.rating(mu=r.crewmate_mu, sigma=r.crewmate_sigma)
                crewmate_data.append((p, r, os_r))

        if not impostor_data or not crewmate_data:
            continue

        impostor_team = [d[2] for d in impostor_data]
        crewmate_team = [d[2] for d in crewmate_data]

        def create_meta(team: list) -> object:
            avg_mu = sum(r.mu for r in team) / len(team)
            avg_var = sum(r.sigma**2 for r in team) / len(team)
            return RATING_MODEL.rating(mu=avg_mu, sigma=math.sqrt(avg_var))

        meta_imp = create_meta(impostor_team)
        meta_crew = create_meta(crewmate_team)

        ranks = [0, 1] if impostors_won else [1, 0]
        new_imp, new_crew = RATING_MODEL.rate([[meta_imp], [meta_crew]], ranks=ranks)

        imp_mu_delta = new_imp[0].mu - meta_imp.mu
        crew_mu_delta = new_crew[0].mu - meta_crew.mu
        imp_sigma_ratio = new_imp[0].sigma / meta_imp.sigma
        crew_sigma_ratio = new_crew[0].sigma / meta_crew.sigma

        imp_total_var = sum(r.sigma**2 for r in impostor_team)
        imp_pool = imp_mu_delta * len(impostor_team)
        for _, rating, old_r in impostor_data:
            share = old_r.sigma**2 / imp_total_var
            rating.impostor_mu = old_r.mu + (imp_pool * share)
            rating.impostor_sigma = max(0.1, old_r.sigma * imp_sigma_ratio)
            rating.impostor_games += 1
            if impostors_won:
                rating.impostor_wins += 1

        crew_total_var = sum(r.sigma**2 for r in crewmate_team)
        crew_pool = crew_mu_delta * len(crewmate_team)
        for _, rating, old_r in crewmate_data:
            share = old_r.sigma**2 / crew_total_var
            rating.crewmate_mu = old_r.mu + (crew_pool * share)
            rating.crewmate_sigma = max(0.1, old_r.sigma * crew_sigma_ratio)
            rating.crewmate_games += 1
            if not impostors_won:
                rating.crewmate_wins += 1

    return _build_rankings_from_ratings(models, temp_ratings)
