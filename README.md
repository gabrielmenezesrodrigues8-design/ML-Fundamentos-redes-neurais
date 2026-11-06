# Rede Neural Simples com NumPy (2 Camadas)

Este repositório contém a implementação de uma rede neural simples de 2 camadas utilizando apenas a biblioteca **NumPy**. O objetivo deste código é demonstrar, em cerca de 20 linhas, os fundamentos de como uma rede neural aprende.

O código aborda diretamente os conceitos fundamentais de Deep Learning:
* **Forward Pass** (Cálculo da saída)
* **Loss** (Medição do erro)
* **Backward Pass** (Retropropagação)
* **Gradient Descent** (Atualização dos pesos)

---

## 💻 O Código Completo

```python
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```

---

## 🧠 Explicando o Código Linha por Linha

### 1. Definindo as Dimensões e Dados Iniciais

```python
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)
```

* **`N = 64`**: Tamanho do lote (*batch size*). Processamos 64 exemplos de uma vez.
* **`D_in = 1000`**: Número de atributos de entrada (dimensão de cada dado).
* **`H = 100`**: Número de neurônios na camada oculta (*hidden layer*).
* **`D_out = 10`**: Número de saídas previstas pela rede.
* **`w1` e `w2`**: As matrizes de pesos aleatórias que a rede tentará aprender. `w1` conecta a entrada à camada oculta e `w2` conecta a camada oculta à saída.

### 2. Forward Pass (Calculando a Previsão)

```python
h = 1 / (1 + np.exp(-x.dot(w1)))
y_pred = h.dot(w2)
```

* **`h`**: Realiza a multiplicação matricial entre a entrada `x` e os pesos `w1` (`x.dot(w1)`) e aplica a função de ativação Sigmoide: $\sigma(z) = \frac{1}{1 + e^{-z}}$. A variável `h` representa as ativações da camada oculta.
* **`y_pred`**: Multiplica as ativações `h` pela matriz de pesos `w2` para gerar a previsão final da rede.

### 3. Função de Perda (Loss)

```python
loss = np.square(y_pred - y).sum()
print(t, loss)
```

* Calcula o erro quadrático total entre a previsão (`y_pred`) e o valor real esperado (`y`). O objetivo do treino é fazer este valor de *loss* diminuir a cada iteração.

### 4. Backward Pass (Retropropagação dos Gradientes)

Aqui é onde a **Regra da Cadeia** é aplicada do final para o início para descobrir como ajustar cada peso:

```python
grad_y_pred = 2.0 * (y_pred - y)
grad_w2 = h.T.dot(grad_y_pred)
grad_h = grad_y_pred.dot(w2.T)
grad_w1 = x.T.dot(grad_h * h * (1 - h))
```

* **`grad_y_pred`**: Derivada da função de perda em relação à saída `y_pred`
* **`grad_w2`**: Gradiente em relação aos pesos `w2`. Usa a transposta das ativações `h.T` multiplicada pelo gradiente vindo da frente (`grad_y_pred`).
* **`grad_h`**: Propaga o gradiente de volta para a camada oculta atravessando a matriz `w2.T`.
* **`grad_w1`**: Gradiente em relação aos pesos `w1`. O termo `h * (1 - h)` é a derivada exata da função de ativação Sigmoide. Ele multiplica o gradiente recebido da frente (`grad_h`) e combina com a entrada `x.T`.

### 5. Atualizando os Pesos (Gradient Descent)

```python
w1 -= 1e-4 * grad_w1
w2 -= 1e-4 * grad_w2
```

* Subtrai dos pesos uma pequena fração do gradiente (`1e-4` é a taxa de aprendizado ou *learning rate*). Isso ajusta `w1` e `w2` na direção que reduz o erro.
