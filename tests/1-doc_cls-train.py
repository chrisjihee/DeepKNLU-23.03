import os
import sys
from pathlib import Path

from chrisbase.io import get_current_path
from chrisbase.util import to_dataframe
from chrisdict import AttrDict

env = AttrDict()
env["python_path"] = Path(sys.executable)
env["project_path"] = [x for x in get_current_path().parents if x.name.startswith("DeepKorean")][0]
env["current_path"] = get_current_path().relative_to(env.project_path)
os.chdir(env.project_path)


def main():
    print(to_dataframe(env, columns=["key", "value"]))

    ################################################################################
    # 코드 4-4: 모델 환경 설정
    ################################################################################
    from ratsnlp.nlpbook.classification import ClassificationTrainArguments

    args = ClassificationTrainArguments(
        pretrained_model_name="pretrained/KcBERT-Base",
        downstream_corpus_name="nsmc",
        downstream_corpus_root_dir="data",
        downstream_model_dir="checkpoints/nsmc",
        batch_size=32,
        learning_rate=5e-5,
        max_seq_length=128,
        epochs=3,
        tpu_cores=0,
        seed=7,
        # overwrite_cache=True,
    )
    print(f"args={args}")

    ################################################################################
    # 코드 4-5: 랜덤 시드 고정
    ################################################################################
    from ratsnlp import nlpbook

    nlpbook.set_seed(args)

    ################################################################################
    # 코드 4-6: 로거 설정
    ################################################################################
    nlpbook.set_logger(args)

    ################################################################################
    # 코드 4-7: 말뭉치 내려받기
    ################################################################################
    from Korpora import Korpora

    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        # force_download=True
    )

    ################################################################################
    # 코드 4-8: 토크나이저 준비
    ################################################################################
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    print(f"tokenizer={tokenizer}")
    print(f'tokenizer.tokenize("안녕하세요. 반갑습니다.")={tokenizer.tokenize("안녕하세요. 반갑습니다.")}')

    ################################################################################
    # 코드 4-9: 학습 데이터셋 구축
    ################################################################################
    from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset

    corpus = NsmcCorpus()
    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )

    ################################################################################
    # 코드 4-10: 학습 데이터 로더 구축
    ################################################################################
    from torch.utils.data import DataLoader, RandomSampler

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    ################################################################################
    # 코드 4-11: 평가용 데이터 로더 구축
    ################################################################################
    from torch.utils.data import SequentialSampler

    val_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )

    ################################################################################
    # 코드 4-12: 모델 초기화
    ################################################################################
    from transformers import BertConfig, BertForSequenceClassification

    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )

    ################################################################################
    # 코드 4-13: TASK 정의
    ################################################################################
    from ratsnlp.nlpbook.classification import ClassificationTask

    task = ClassificationTask(model, args)

    ################################################################################
    # 코드 4-14: 트레이너 정의
    ################################################################################
    trainer = nlpbook.get_trainer(args)

    ################################################################################
    # 코드 4-15: 학습 개시
    ################################################################################
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == '__main__':
    main()
