<div align="center">
    <h1>
    RoboOmni: Proactive Robot Manipulation in Omni-modal Context
    </h1>
</div>


<p align="center">
  📖 <a href="https://arxiv.org/abs/xxxxxxxxxx"><strong>arXiv Paper</strong></a> |   
  🌐 <a href="https://sinwang20.github.io/RoboOmni"><strong>Website</strong></a> | 
  🤗 <a href="https://huggingface.co/sinwang20/"><strong>Model</strong></a> | 
  🤗 <a href="https://huggingface.co/sinwang20/"><strong>Dataset</strong></a> | 
  🛠️ <a href="https://github.com/sinwang20/RoboOmni"><strong>Github</strong></a> | 
</p>

![logo](./assets/logo.png)


<!-- # RoboOmni -->

<!-- ![Omni Logo](./assets/logo.png) -->
---

Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision–Language–Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively.
In this work, we introduce *cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands.* To address this new setting, we present **RoboOmni**, a *Perceiver-Thinker-Talker-Executor* framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. 
To address the absence of training data for proactive intention recognition in robotic manipulation, we build **OmniAction** comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.

---


<video controls>
  <source src="./assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## ⭐️ Architecture

At the heart of RoboOmni lies the Perceiver-Thinker-Talker-Executor architecture, which unifies multiple modalities (vision, speech, environmental sounds) into a single, seamless framework for robot action execution.

![Omni Logo](./assets/model.jpg)


## 🤗 Model & Datasets Zoo


| Models               | Checkpoint                                                     | Description                                                | 
|----------------------|----------------------------------------------------------------|------------------------------------------------------------|
| RoboOmni Base     | [🤗 sinwang20/RoboOmni](https://huggingface.co/sinwang20/RoboOmni)   | Base model of the RoboOmni               | 
| RoboOmni-LIBERO     | [🤗 sinwang20/RoboOmni-LIBERO](https://huggingface.co/sinwang20/RoboOmni-LIBERO)   | Finetuned model on OmniAction-LIBERO based on RoboOmni Base                 | 


| Dataset               | Checkpoint                                                     | Description                                                | 
|----------------------|----------------------------------------------------------------|------------------------------------------------------------|
| OmniAction     | [🤗 sinwang20/OmniAction](https://huggingface.co/sinwang20/OmniAction)   | 140k trajectory               | 
| OmniAction-LIBERO     | [🤗 sinwang20/OmniAction-LIBERO](https://huggingface.co/sinwang20/OmniAction-LIBERO)   | Training and evaluation data on  OmniAction-LIBERO benchmark                | 


## 📍 Getting Started

```bash
cd training

# Create and activate conda environment
conda create -n roboomni python=3.10 -y
conda activate roboomni

# Install dependencies
pip install -r requirements.txt

```


## 📦 Pretraining

### OmniAction Dataset

![Omni Logo](./assets/omni.png)

We introduce OmniAction, a large-scale multimodal dataset for contextual instruction following. It comprises 141,162 episodes covering 112 skills and 748 objects, enriched with 5,096 distinct speaker timbres, 2,482 non-verbal sound events, and 640 environmental backgrounds. The dataset spans six categories of contextual instructions—sentiment cues, overlapping voices, non-verbal cues, identity cues, dyadic dialogue, and triadic dialogue—capturing both subtle affective signals and complex multi-party interactions in everyday settings.

![Omni Logo](./assets/data.jpg)


- **Format**: RLDS (Reinforcement Learning Datasets standard).  
- **Audio**: Sorted according to filename.  

Place your data in the following structure:

```
./data/audio/ # audio files
./data/omniaction/ # RLDS-formatted OmniAction dataset
```

### Run Pretraining

Use the provided script:

```bash
bash pretrain.sh 
```

The following variants should be set:
- `output_dir`: path to save checkpoints and logs.
- `resume_from_checkpoint`: resume training from a checkpoint (optional).
- `data_root_dir`: root directory of your data.
- `data_mix`: choose data mixture, e.g. omniaction.



## 🔧 Finetuning

For the finetuning RoboOmni base model on OmniAction-LIBERO benchmarks.

### Run Finetuning

```
bash train_libero.sh
```
The following variants should be set:
- `output_dir`: path to save checkpoints and logs.
- `resume_from_checkpoint`: resume training from a checkpoint (optional).
- `data_root_dir`: root directory of your data.
- `data_mix`: choose data mixture. (Avaliable options: ['omniaction_libero_spatial', 'omniaction_libero_goal', 'omniaction_libero_object', 'omniaction_libero10'])


### Evaluation on OmniAction-LIBERO


```
python experiments/libero/run_libero_eval.py \
  --pretrained_checkpoint MODEL_PATH \
  --task_suite_name libero_spatial \
  --speech overlap
```
- Supported `--task_suite_name` types: ['libero_spatial', 'libero_object', 'libero_goal', 'libero-10']
- Supported `--speech` types: ['overlap', 'identity', 'sentiment', 'events', 'two', 'three']


## 👋 Citation

**BibTeX:**

```bibtex
@article{wang2025roboomni,
  title={RoboOmni: Proactive Robot Manipulation in Omni-modal Context},
  author={Siyin Wang and Jinlan Fu and Feihong Liu and Xinzhe He and Huangxuan Wu and Junhao Shi and Kexin Huang and Zhaoye Fei and Jingjing Gong and Zuxuan Wu and Yugang Jiang and See-Kiong Ng and Tat-Seng Chua and Xipeng Qiu},
  journal={arXiv preprint arXiv:2510.xxxx},
  year={2025},
  url={https://arxiv.org/abs/2410.xxxx},
  archivePrefix={arXiv},
  eprint={2510.xxxx},
  primaryClass={cs.RO},
}
```