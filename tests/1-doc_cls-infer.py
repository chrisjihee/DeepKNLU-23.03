from chrislab.ratsnlp import cli
from chrislab.common.util import GpuProjectEnv
from ratsnlp.nlpbook.classification import ClassificationDeployArguments

with GpuProjectEnv(project_name="DeepKorean", working_gpus="0") as env:
    args = ClassificationDeployArguments(
        working_config_file=env.running_file.with_suffix('.json').name,
        pretrained_model_path="model/pretrained/KcBERT-Base",
        downstream_model_path="model/finetuned/nsmc (dl012) [03.20 21:34:29]",
        downstream_model_file=None,
        max_seq_length=128,
    )
    config = args.save_working_config()
    assert config.exists(), f"No config file: {config}"

print(f"config={config}")
print(f"env={env}")
exit(1)

################################################################################
# 코드4 토크나이저 로드
################################################################################
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_path,
    do_lower_case=False,
)
print(f"tokenizer={tokenizer}")
print(f'tokenizer.tokenize("안녕하세요. 반갑습니다.")={tokenizer.tokenize("안녕하세요. 반갑습니다.")}')

################################################################################
# 코드5 체크포인트 로드
################################################################################
import torch

fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)

################################################################################
# 코드6 BERT 설정 로드
################################################################################
from transformers import BertConfig

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_path,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)

################################################################################
# 코드7 BERT 모델 초기화
################################################################################
from transformers import BertForSequenceClassification

model = BertForSequenceClassification(pretrained_model_config)

################################################################################
# 코드8 체크포인트 읽기
################################################################################
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})

################################################################################
# 코드9 평가 모드 전환
################################################################################
model.eval()


################################################################################
# 코드10 INFERENCE
################################################################################
def inference_fn(sentence):
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
        prob = outputs.logits.softmax(dim=1)
        positive_prob = round(prob[0][1].item(), 4)
        negative_prob = round(prob[0][0].item(), 4)
        pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
    return {
        'sentence': sentence,
        'prediction': pred,
        'positive_data': f"긍정 {positive_prob}",
        'negative_data': f"부정 {negative_prob}",
        'positive_width': f"{positive_prob * 100}%",
        'negative_width': f"{negative_prob * 100}%",
    }


################################################################################
# 코드11 웹 서비스
################################################################################
from ratsnlp.nlpbook.classification import get_web_service_app

app = get_web_service_app(inference_fn)
app.run()
