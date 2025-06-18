Use at your own risk. Written with Gemini, Claude and OpenAI under my prompting.  They  work on my RTX 4090 windows 11 machine under WSL2


# quickSegMatte
Various scripts to procedurally generate mattes with AI segmenting


## multi_object_remover4.py


Procedurally generate segmentation mattes for VFX. This script automatically remove people using DeepLabV3, and optionally uses CLIPseg to remove any objects specified. There is a thresho;d for each object and a negative prompt if something is getting matted which shouldn't be.


### setup and install dependencies
```
pyenv virtualenv segment
pyenv local segment

pip install torch torchvision pillow tqdm transformers diffusers accelerate

python multi_object_remover4.py --help
 
usage: multi_object_remover4.py [-h] [--prompts [PROMPTS ...]] [--thresholds [THRESHOLDS ...]]
                                [--negative-prompts [NEGATIVE_PROMPTS ...]]
                                [--negative-thresholds [NEGATIVE_THRESHOLDS ...]] --input-folder INPUT_FOLDER
                                --output-folder OUTPUT_FOLDER

Two-stage pipeline: auto-remove people using , then remove custom objects.

options:
  -h, --help            show this help message and exit
  --prompts [PROMPTS ...]
                        [Stage 2] Objects to remove (e.g., 'a garden hose').
  --thresholds [THRESHOLDS ...]
                        [Stage 2] Threshold for each positive prompt.
  --negative-prompts [NEGATIVE_PROMPTS ...]
                        [Stage 2] Objects to protect from removal.
  --negative-thresholds [NEGATIVE_THRESHOLDS ...]
                        [Stage 2] Threshold for each negative prompt.
  --input-folder INPUT_FOLDER
                        Path to input images.
  --output-folder OUTPUT_FOLDER
                        Path for output images.

```
### examples

Remove people:
```
python multi_object_remover4.py --input-folder in_dir --output-folder out_dir
```
remove people, remove what your kinda think is a beach ball and what you are pretty sure is umbrellas, make sure sand is not incorrectly removed:
```
python multi_object_remover4.py --input-folder in_dir --output-folder out_dir --prompts "beach ball" "umbrella" --thresholds .65 .85  --negative-promps "sand" --negative-thresholds .8
```
