# Positional Encoding

Postitional Encoding은 다음과 같은 방식을 사용한다.

$$
\begin{align}
PE(pos, 2i) &= \sin(pos/10000^{2i/d_{nodel}})
\\
PE(pos, 2i+1) &= \cos(pos/10000^{2i/d_{nodel}})
\end{align}
$$  

여기서 pos란 문장의 단어 개수, $d_{model}$은 워드 벡터가 가지는 차원 수 이다.  


# Mini Batch

Mini Batch는 다수의 데이터를 한번에 계산하여 신경망을 학습시키는 방법론이다. Mini Batch로 학습을 시키기 위해서는 학습 데이터의 크기 (Channel 등)이 모두 같아야 하며 Batch의 크기는 GPU 사용량에 비례한다.

만약 각 Stage 마다 Patch를 쪼갤 때 쪼개는 개수가 다르다면 차원의 수가 달라지기 때문에 Mini Batch 학습이 불가능 하다. 따라서 attention map을 보면서 threshold를 지정하는것이 아니라, 각 Stage마다 쪼개야 하는 patch의 개수를 정해야만 학습할수 있다.  