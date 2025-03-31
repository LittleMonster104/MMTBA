# TEACH
A Multi-Modal Dataset for Analyzing Teacher Behavior

## Environment
```bash
Linux: Ubuntu 22.04
python: 3.10
GPU: A6000(48G)
```

## Teaching Action Detection 

```bash
conda create -n tea_ac python=3.10.13
conda activate tea_ac
cd TEACH/Teaching_Action_Detection
pip install -r requirements.txt
```

```bash
# The model training and testing are based on the mmaction2 framework.(https://github.com/open-mmlab/mmaction2)
cd my_mmaction

# train
python tools/train.py configs/detection/acrn/my-slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-20e_ava21-rgb.py --validate
# Different models and methods can be used for training by changing the path of the configuration file. The trained weight files will be saved in the work_dirs directory.

#test
python tools/test.py configs/detection/acrn/my-slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-20e_ava21-rgb.py work_dirs/my-slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-20e_ava21-rgb/best_mAP_overall_epoch_9.pth
# The paths for the configuration files and weight files can be customized and replaced.

#demo
python demo/demo_spatiotemporal_det.py demo/demo_input/demo.mp4 demo/demo_output/det_demo.mp4 --config configs/detection/acrn/my-slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-20e_ava21-rgb.py --checkpoint work_dirs/my-slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-20e_ava21-rgb/best_mAP_overall_epoch_9.pth  --det-checkpoint Checkpionts/mmdetection/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth --det-score-thr 0.5 --action-score-thr 0.5 --output-stepsize 4  --output-fps 6 --label-map tools/data/ava/label_map.txt
```

Main Annotation File Description:
```bash
train.csv：Core training file, including filename, frame number, person bounding box coordinates, label, and person ID.
dense_proposals_train.pkl：Proposals feature file.
action_list.pbtxt: Action label classification.
included_timestamps.txt: Included video frame number.
train_excluded_timestamps.csv：Video frames to be excluded.
# The other files mainly consist of intermediate code and results from dataset processing.
```

## Teaching Lecture Evaluation 

Fine-tuning the Baichuan2 large language model：
```bash
conda create -n lec_eva python=3.10.13
conda activate lec_eva
cd TEACH/Teaching_Lecture_Evaluation/
pip install -r requirements.txt
```

```bash
cd baichuan2/Baichuan2-main/fine-tune
# Edit `train.sh` to customize the save path for the parameter files:
--output_dir "/pth/output"
# Fine-tune the training
sh train.sh

# Generate the report
# Change the `lora_path` in `gene_report.py` to the custom path of the parameter files
lora_path = '/pth/output'
# Run gene_report.py
python gene_report.py 
# If you encounter file path issues, you can modify it to an absolute path.
```

File Description:
```bash
teacher_lecture_audios：Lecture audio files (de-identified)
teacher_lecture_texts：Lecture speech-to-text files (de-identified)
gpt_report：Files related to generating evaluation reports using the GPT-3.5 API
finetune_data：Data used for fine-tuning other large language models
```



We also tested fine-tuning using other models, implemented through the LLaMA-Factory tool.(https://github.com/hiyouga/LLaMA-Factory)
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install llamafactory[metrics]==0.7.1
pip install accelerate==0.30.1

cd LLaMA-Factory
#Launch the WebUI and access it via the provided link.
export USE_MODELSCOPE_HUB=1 && llamafactory-cli webui
#Move 'train.json' and 'eval.json' from the 'finetune_data' directory to the 'data' directory, and modify 'datainfo.json'.
#For WebUI page operations, refer to this link(https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory) to perform model fine-tuning and dialogues.
```



## Instructional Design 

File Descriptions:
```bash
docx files: Teacher instructional designs
speech_text.xlsx: Teacher lecture speech-to-text files
instruction_design_anno.xlsx: Instructional design and teaching process annotation files
```

## Dataset Details

### Metadata
```bash
Gender(0:male，1:female),
Grade(eg: p1: First grade of elementary school, j1: First grade of junior high school, s2: Second grade of senior high school),
Teaching content,
Teaching Experience (eg: 1y: one year, 2y: two years, 6m: six months),
Teacher Qualification Certificate (0: not owned, 1: owned)
```

### Data Anonymization
```bash
For the teaching video footage, we used facial masking to prevent the disclosure of personal privacy and information.
In the teacher lecture content, we employed muting and trimming methods to remove sensitive information.
For textual content, we conducted manual reviews to eliminate personal information such as references to schools and names.
```

### Potential Biases
The dataset may contain potential biases related to time periods, subjects, and teachers.  We encourage community feedback to help identify and improve any bias issues that may exist in the dataset.

## Acknowledgement
This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [mmaction2](https://github.com/open-mmlab/mmaction2). Thanks for their wonderful works.
