from setuptools import setup, find_packages

setup(
    name="hybridmind",
    version="1.0.0",
    description="Vector + Graph Native Database for AI Retrieval",
    author="CodeHashira Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.0",
        "networkx>=3.2.0",
        "aiosqlite>=0.19.0",
        "typer>=0.9.0",
        "rich>=13.7.0",
        "python-multipart>=0.0.6",
        "orjson>=3.9.0",
    ],
    extras_require={
        "ui": ["streamlit>=1.28.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hybridmind=cli.main:app",
        ],
    },
)

