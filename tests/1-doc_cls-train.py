import torch
from Korpora import Korpora
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler

from chrisbase.io import JobTimer, out_hr
from chrisbase.util import to_dataframe
from chrislab.common.util import GpuProjectEnv
from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import ClassificationTask
from ratsnlp.nlpbook.classification import ClassificationTrainArguments
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer

env = GpuProjectEnv(project_name="DeepKorean", working_gpus="0")
print(to_dataframe(env))
out_hr(c='-')

args = ClassificationTrainArguments(
    pretrained_model_name="pretrained/KcBERT-Base",
    downstream_corpus_root_dir="data",
    downstream_corpus_name="nsmc",
    downstream_model_dir="checkpoints/nsmc",
    monitor="max val_acc",
    learning_rate=5e-5,
    max_seq_length=128,
    batch_size=200,
    save_top_k=2,
    epochs=3,
    seed=7,
)
print(to_dataframe(args))
out_hr(c='-')

with JobTimer(f"{env.project_name}(finetuning {args.pretrained_model_name} using {args.downstream_corpus_name} data and {env.number_of_gpus} devices)",
              mt=1, mb=1, rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
    nlpbook.set_seed(args)
    nlpbook.set_logger()
    out_hr(c='-')

    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
    )
    out_hr(c='-')

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    print(f"tokenizer={tokenizer}")
    print(f"tokenized example={tokenizer.tokenize('안녕하세요. 반갑습니다.')}")
    out_hr(c='-')

    corpus = NsmcCorpus()
    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    out_hr(c='-')

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
    out_hr(c='-')

    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )
    out_hr(c='-')

    torch.set_float32_matmul_precision('high')
    nlpbook.get_trainer(args).fit(
        ClassificationTask(model, args),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
