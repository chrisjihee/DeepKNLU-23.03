mkdir -p model/pretrained
cd model/pretrained

git clone https://huggingface.co/bert-base-multilingual-uncased           Google-BERT-Base
git clone https://huggingface.co/klue/bert-base                           KLUE-BERT-Base
git clone https://huggingface.co/klue/roberta-base                        KLUE-RoBERTa-Base
git clone https://github.com/KPFBERT/kpfbert                              KPF-BERT-Base
git clone https://huggingface.co/beomi/kcbert-base                        KcBERT-Base
git clone https://huggingface.co/beomi/KcELECTRA-base-v2022               KcELECTRA-Base
git clone https://huggingface.co/skt/kobert-base-v1                       KoBERT-Base
git clone https://huggingface.co/monologg/kobigbird-bert-base             KoBigBird-Base
git clone https://huggingface.co/monologg/koelectra-base-v3-discriminator KoELECTRA-Base

cd Google-BERT-Base; rm -rf .* flax_model.* tf_model.*; cd ..;
cd KLUE-BERT-Base; rm -rf .*; cd ..;
cd KLUE-RoBERTa-Base; rm -rf .*; cd ..;
cd KPF-BERT-Base; rm -rf .* tf_model.*; cd ..;
cd KcBERT-Base; rm -rf .*; cd ..;
cd KcELECTRA-Base-v2022; rm -rf .*; cd ..;
cd KoBERT-Base; rm -rf .*; cd ..;
cd KoBigBird-Base; rm -rf .*; cd ..;
cd KoELECTRA-Base-v3; rm -rf .*; cd ..;

