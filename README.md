# miniGPT
Train gpt2(small) from reproducing it.

## Clone Repository
```bash
git lfs clone git@github.com:Zhuangjizzz/miniGPT.git
```
## Install 
### 1.Init Conda
```bash
conda init
conda create -n miniGPT python=3.11
conda activate miniGPT
```
### 2.Install requirements
install with requirements.txt
```bash
pip install -r requirements.txt
```
install with pip command
```bash
pip install torch transformers numpy tiktoken datasets tqdm
```

## quick test
We have trained gpt2 using OpenWebText dataset with `NVIDIA GeForce RTX 3090` for one week,
 the model is saved in `output/trained_gpt_model.pth`,you can try to run sample.py to generate some text based on the model.
The following command will generate text with the `Once upon a time` as the input.
```bash
python sample.py
```
The output looks like this:
```bash
输入提示： Once upon a time
生成文本： Once upon a time, you take that protagonist as a Chief Servant and attack him, making him fall back in his cell to accomplish something (this will always cause him to fail his crime) and immediately complete your quest. * Run from him whilst he rides a massive candy wagon (uses up his rapier and charges backwards. Guides Ty like a cover character and must charge at him) then turn around in his brain to open doors that flicked open into your game. He'll carry you across the castle and end upon your death, sealing your fate forever. Naturally, whenever this unlockable Achievement is unlocked, the player must use the Mechanican Beerfully to infect you with own Undead Constructile. This has sprung into beating Column Bosses and Armors, and it pays off. Use Sick to heal and shock zombies, and Mansion Wriggul can find them poking around chests and battling each other. In order to revive them, use SickSpider's voodoo missiles to the face, punting each vile
```
If you want to try with your own model or generate text with your own input, you can input the following command.
```bash
python sample.py --checkpoint_path your_saved_model.pth --prompt your_input_text
```

## Train the model
### Train gpt2 small from scratch
This will train GPT2 using the `wikitext-103-v1` dataset as default, you can choose your own dataset by modifying prepare.py
We also provide `OpenWebText` dataset, which contains ~ 9B tokens in train.bin, 4M tokens in valid.bin.By modifying prepare.py you can choose to use `wikitext-103-v1` or `OpenWebText` dataset.
```bash
python train.py --init_from resume --checkpoint_path output/trained_gpt_model.pth
```
then you can run sample.py to generate some text based on the model.
```bash
python sample.py --checkpoint_path your_saved_model.pth
```


