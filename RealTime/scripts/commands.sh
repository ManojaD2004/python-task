Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

python -m  venv deps

./deps/Scripts/activate.ps1

docker build -t model-py -f dockerfile.python ./

docker run -p 5222:5222 -v "${PWD}/public/images:/app/images" -v "${PWD}/model_data:/app/model_data" -e PYTHONUNBUFFERED=1 --rm model-py python main_video2.py /app/model_data/demo.json http://192.168.1.7:4747/video 1 5 3 localhost:29092

docker run -p 5222:5222 -v "${PWD}/images:/app/images" -v "${PWD}/model_data:/app/model_data" -e PYTHONUNBUFFERED=1 --rm model-py-2 python main_video2.py /app/model_data/demo.json http://192.168.1.7:4747/video 1 5 3 localhost:29092

python ./scripts/run.py ./model_data/demo.json rtsp://test123:test123@192.168.1.8/stream1 1 5 3 localhost:29092

python ./scripts/run.py ./model_data/demo.json rtsp://test123:test123@192.168.137.183/stream1 1 5 1 localhost:29092