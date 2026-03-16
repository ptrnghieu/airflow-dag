"""
Pydantic models for data validation
"""
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional
from datetime import datetime


class ArxivPaper(BaseModel):
    """
    Pydantic model for ArXiv paper data.
    Ensures data consistency and validation before saving to MongoDB.
    """
    id: str = Field(..., description="ArXiv paper ID")
    title: str = Field(..., description="Paper title", min_length=1)
    authors: str = Field(..., description="Paper authors", min_length=1)
    abstract: Optional[str] = Field(None, description="Paper abstract")
    published: str = Field(..., description="Publication date (YYYY-MM-DD)")
    updated: str = Field(..., description="Last update date (YYYY-MM-DD)")
    categories: str = Field(..., description="Paper categories")
    pdf_url: str = Field(..., description="PDF URL")
    doi: Optional[str] = Field(None, description="DOI")
    comment: Optional[str] = Field(None, description="Paper comment")
    journal_ref: Optional[str] = Field(None, description="Journal reference")
    scraped_at: str = Field(..., description="Scraping timestamp")
    data_quality: str = Field(default="good", description="Data quality flag")

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate paper ID is not empty"""
        if not v or v.strip() == '':
            raise ValueError('Paper ID cannot be empty')
        return v.strip()

    @field_validator('title', 'authors')
    @classmethod
    def validate_required_strings(cls, v: str) -> str:
        """Validate required string fields"""
        if not v or v.strip() == '':
            raise ValueError('This field cannot be empty')
        return v.strip()

    @field_validator('published', 'updated')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @field_validator('pdf_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

    @field_validator('data_quality')
    @classmethod
    def validate_quality(cls, v: str) -> str:
        """Validate data quality flag"""
        valid_qualities = ['good', 'missing_abstract', 'incomplete']
        if v not in valid_qualities:
            raise ValueError(f'Data quality must be one of: {valid_qualities}')
        return v

    class Config:
        """Pydantic configuration"""
        # Allow extra fields to be ignored
        extra = 'ignore'
        # Use str for validation errors
        str_strip_whitespace = True
        # Validate on assignment
        validate_assignment = True


class ArxivPaperList(BaseModel):
    """
    Model for a list of ArXiv papers.
    Used for batch validation.
    """
    papers: list[ArxivPaper] = Field(..., description="List of papers")
    total_count: int = Field(..., description="Total number of papers", ge=0)
    
    @field_validator('papers')
    @classmethod
    def validate_papers(cls, v: list[ArxivPaper]) -> list[ArxivPaper]:
        """Validate papers list is not empty"""
        if not v:
            raise ValueError('Papers list cannot be empty')
        return v

    @field_validator('total_count')
    @classmethod
    def validate_count(cls, v: int, info) -> int:
        """Validate total count matches papers list length"""
        papers = info.data.get('papers', [])
        if v != len(papers):
            raise ValueError(f'Total count ({v}) does not match papers list length ({len(papers)})')
        return v

    class Config:
        """Pydantic configuration"""
        extra = 'ignore'
        validate_assignment = True

