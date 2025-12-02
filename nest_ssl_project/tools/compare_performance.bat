@echo off
REM Benchmark script to compare NeMo vs local encoder performance

echo Running encoder performance benchmark...
python nest_ssl_project/tools/benchmark_encoder.py --compare --num_iter 100

echo.
echo For detailed profiling, run:
echo python nest_ssl_project/tools/benchmark_encoder.py --compare --profile

pause

