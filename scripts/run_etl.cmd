@echo off
setlocal ENABLEDELAYEDEXPANSION

:: Use UTF-8 console to avoid Unicode logging errors
chcp 65001 >NUL 2>&1
set PYTHONIOENCODING=utf-8

:: Change to project root (this script is inside scripts\)
pushd "%~dp0.."

:: Optional: activate conda env if available
where conda >NUL 2>&1
if %ERRORLEVEL% EQU 0 (
  echo Activating conda environment: superml
  call conda activate superml
) else (
  echo conda not found in PATH. Continuing with default Python.
)

:: Ensure local SQLite DB is used for development
set DATABASE_URL=sqlite:///data/customer_insights.db

:: Create data directory if missing
if not exist data mkdir data

echo Running DB setup (ETL 1)...
python scripts\setup_database.py || goto :error

echo Generating sample data (ETL 2)...
python scripts\generate_standalone_data.py || goto :error

echo ETL pipelines completed successfully.
popd
endlocal
exit /b 0

:error
echo ETL failed. See error above.
popd
endlocal
exit /b 1
