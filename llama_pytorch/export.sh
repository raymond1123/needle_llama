
CONVERT_NAME='story45.bin'
MODEL_PATH='/home/raymond/workspace/mmmove/llm/llama/needle_llama/llama_pytorch/story_model'

python export.py ${CONVERT_NAME} \
    --meta-llama ${MODEL_PATH}

