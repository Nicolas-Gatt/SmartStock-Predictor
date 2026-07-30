#  SmartStock-Predictor

Um pipeline de Machine Learning em Python orientado a objetos para previsão de demanda e faturamento no varejo utilizando dados históricos da rede Walmart. O diferencial da aplicação é converter o erro estatístico do modelo em métricas financeiras reais, gerando automaticamente um orçamento de compras e calculando custos operacionais de estoque.

##  Funcionalidades

* **Análise Financeira Assimétrica:** Avalia os custos operacionais diferenciando capital imobilizado por excesso de mercadoria (*Overstock*) de lucro não realizado por falta de produtos nas prateleiras (*Stockout*).
* **Orçamento Inteligente de Estoque:** Aplica regras de negócio para calcular o Custo da Mercadoria Vendida (CMV) e recomendar o aporte exato de capital para aquisição de inventário com 94% de acurácia.
* **Engenharia de Recursos Temporais:** Extração e sanitização automática de componentes de sazonalidade (ano, mês, semana do ano), com tratamento robusto para datas inválidas e prevenção de vazamento de dados.
* **Arquitetura Limpa e Orientada a Objetos:** Estrutura desenvolvida sob os princípios de *Clean Code*, SOLID (SRP) e padrão *Facade*, isolando regras financeiras da modelagem estatística.
* **Observabilidade de Produção:** Sistema nativo de *logging* integrado que exibe o progresso de treinamento nos núcleos da CPU, métricas estatísticas (RMSE) e o relatório executivo de gastos no terminal.

##  Tecnologias Utilizadas

* Python 3.8+
* Scikit-Learn (Modelagem preditiva com *RandomForestRegressor*)
* Pandas & NumPy (Manipulação de dados e álgebra linear)

## Como Executar o Projeto Localmente

Certifique-se de que o arquivo `Walmart_Sales.csv` esteja salvo dentro da pasta `data/`. Siga os comandos abaixo no terminal para configurar e rodar a aplicação em sua máquina:

```bash
# 1. Crie o ambiente virtual
python3 -m venv venv

# 2. Ative o ambiente virtual
source venv/bin/activate  # No Windows: .\venv\Scripts\Activate.ps1

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Entre na pasta de código e execute o pipeline
cd src
python app.py
