This is the official code for "Adversarially Pretrained Transformers may be Universally Robust In-Context Learners" [S. Kumano et al., preprint].

All generated data can be downloaded from [here](https://filedn.com/lAlreeY65CBjFVbAkaD5F7k/Research/Adversarially%20Pretrained%20Transformers%20may%20be%20Universally%20Robust%20In-Context%20Learners/data.zip) (400KB).


# Setup
```console
docker-compose -f "docker/docker-compose.yaml" up -d --build 
```

# Run
```console
bash/train.sh <gpu_id: int>
bash/test_train.sh <gpu_id: int>
bash/test_normal.sh <gpu_id: int>
bash/test_real.sh <gpu_id: int>
```