from pydantic import BaseModel, Field, validator
from typing import List, Optional
import datetime

class ResearchPaper(BaseModel):
    """
    Data contract for a research paper record.
    Ensures that all necessary fields are present and correctly formatted.
    """
    id: str = Field(..., description="Unique identifier for the paper (e.g., ArXiv ID)")
    title: str = Field(..., min_length=5, description="Title of the research paper")
    abstract: str = Field(..., min_length=20, description="Abstract or summary of the paper")
    authors: List[str] = Field(..., description="List of author names")
    published_date: Optional[datetime.date] = Field(None, description="Date of publication")
    categories: List[str] = Field(default_factory=list, description="Categories or tags (e.g., cs.CL, stat.ML)")
    doi: Optional[str] = Field(None, description="Digital Object Identifier if available")
    url: Optional[str] = Field(None, description="URL to the paper PDF or landing page")

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty or just whitespace')
        return v.strip()

    @validator('abstract')
    def abstract_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Abstract cannot be empty or just whitespace')
        return v.strip()

class SearchResult(ResearchPaper):
    """
    Extends ResearchPaper with search-specific metadata.
    """
    score: float = Field(..., description="Main ranking score Used for sorting")
    retrieval_method: str = Field(..., description="Method used for retrieval (e.g., 'semantic', 'keyword', 'hybrid')")
    
    # Optional detailed scores
    semantic_score: Optional[float] = Field(None, description="Cosine similarity from vector search")
    keyword_score: Optional[float] = Field(None, description="Normalized BM25 score")
    rrf_score: Optional[float] = Field(None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(None, description="Cross-Encoder reranking score")
