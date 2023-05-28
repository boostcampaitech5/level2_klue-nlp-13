# level2_klue-nlp-13

## ✅ 목차
[1. 정보](##📜-정보) > [2. 진행 과정](##🗓️-진행-과정) > [3. 팀원](##👨🏼‍💻-팀원) > [4. 역할](##🧑🏻‍🔧-역할) > [5. 디렉토리 구조](##📁-디렉토리-구조) > [6. 프로젝트 구성](##⚙️-프로젝트-구성)

## 📜 정보
- 주제 : 부스트캠프 5기 Level 2 프로젝트 - 문장 내 개체간 관계 추출(KLUE RE)
- 프로젝트 기간 : 2023년 5월 2일 ~ 2023년 5월 18일
- 프로젝트 내용 : Hugging Face의 Pretrained 모델과KLUE RE 데이터셋을 활용해 주어진 subject, object entity간의 30개 중 하나의 relation 예측하는 AI 모델 구축

<br>

## 🗓️ 진행 과정

<img width="959" alt="Screenshot 2023-05-24 at 3 31 48 PM" src=>

<br>

## 👨🏼‍💻 팀원

<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/Yunhee000"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/Yunhee000"/></a>
            <br/>
            <a href="https://github.com/Yunhee000"><strong>김윤희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/8804who"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/8804who"/></a>
            <br/>
            <a href="https://github.com/8804who"><strong>김주성</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/ella0106"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/ella0106"/></a>
            <br/>
            <a href="https://github.com/ella0106"><strong>박지연</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/bom1215"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/bom1215"/></a>
            <br/>
            <a href="https://github.com/bom1215"><strong>이준범</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/HYOJUNG08"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/HYOJUNG08"/></a>
            <br/>
            <a href="https://github.com/HYOJUNG08"><strong>정효정</strong></a>
            <br />
        </td>
    </tr>
</table>
<br>

## 🧑🏻‍🔧 역할

| 이름 | 역할 |
| :----: | --- |
| **김윤희** | 데이터 전처리(entity token), 품사 태깅, 모델 성능 실험 |
| **김주성** | 코드 모듈화, 데이터 증강(RD, MLM),  모델 성능 실험 |
| **박지연** | 코드 모듈화, Wandb 추가, concat entity 실험 |
| **이준범** | EDA, 데이터 증강(Easy Data Aug), 기능 추가 |
| **정효정** | 모델 리서치, 데이터증강(역번역), 데이터 전처리(Typed entity marker (punct)) |

<br>

## 📁 디렉토리 구조

```bash
├── level2_KLUE-nlp-13
|   ├── basecode/ (private)
│   ├── data/ (private)
│   ├── results/ (private)
│   ├── eda/
│   ├── utils/
│   │   ├── DataAugmentation.py
│   │   ├── DataLoader.py
│   │   ├── Dataset.py
│   │   ├── DataPreprocessing.py
│   │   ├── Inference.py
│   │   ├── Model.py
│   │   ├── R_RoBERTa_model.py
│   │   ├── Score.py
│   │   ├── Train.py
│   │   ├── Utils.py
│   │   ├── backtranslation.ipynb
│   │   ├── customModel.py
│   │   ├── dict_label_to_num.pkl
│   │   └── dict_num_to_label.pkl
│   ├── main.py
│   ├── README.md
│   ├── config.yaml
|   ├── sweep.py
│   └── data_processing.py
```
<br>

## ⚙️ 프로젝트 구성

|분류|내용|
|:--:|--|
|모델|[`klue/bert-base`](https://huggingface.co/klue/bert-base), [`klue/roberta-large`](https://huggingface.co/klue/roberta-large) `HuggingFace Transformer Model`+`Pytorch Lightning`활용 + Attention Layer or FC Layer|
|전처리|• `Entity Representation` : Entity marker / Typed entity marker / SUB,OBJ marker / punct(한글) 등 다양한 entity representation을 적용하여 최적의 성능을 내는 entity representation 적용 |• Evaluation 단계의 피어슨 상관 계수를 일차적으로 비교<br>• 기존 SOTA 모델과 성능이 비슷한 모델을 제출하여 public 점수를 확인하여 이차 검증|
|데이터|• `raw data` : 기본 train 데이터 32470개 <br>• `증강데이터` : |
|검증 전략|• 만들었던 모델의 Validation 데이터를 inference에 Micro F1-Score와 AUPRC Score 비교|
|앙상블 방법| |
|모델 평가 및 개선||

<br>
