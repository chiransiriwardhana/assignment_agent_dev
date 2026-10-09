import re
from datetime import date

def resolve_relative_dates(question: str, today: date | None = None) -> str:
    """'last year' -> explicit previous calendar year, so the LLM never has to guess."""
    y = (today or date.today()).year - 1
    return re.sub(r"\b(?:the )?(?:last|previous) year\b",
                  f"calendar year {y} (date_from={y}-01-01, date_to={y}-12-31)", question, flags=re.I)