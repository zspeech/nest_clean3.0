@echo off
call D:\anaconda\Scripts\activate.bat nemo
python nest_ssl_project/tools/trace_encoder_input_source.py --nemo_dir ./saved_nemo_outputs --nest_dir ./saved_nest_outputs
pause

