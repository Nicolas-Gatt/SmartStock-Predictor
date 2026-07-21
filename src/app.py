# -*- coding: utf-8 -*-
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Configura o logging para exibir o progresso e relatórios no terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constantes do Sistema e Parâmetros de Negócio
DEFAULT_DATASET_PATH: str = "../data/Walmart_Sales.csv"
TARGET_COLUMN: str = "Weekly_Sales"
FEATURE_COLUMNS: List[str] = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
]
TEST_SIZE_RATIO: float = 0.2
RANDOM_SEED: int = 42
COGS_RATE: float = 0.75  # Custo da Mercadoria Vendida (75% da receita vira gasto de estoque)


class FinancialImpactAnalyzer:
    """Traduz previsões de Machine Learning em métricas de custo de estoque e impacto financeiro."""

    def __init__(self, cogs_rate: float = COGS_RATE) -> None:
        self.cogs_rate = cogs_rate

    def evaluate_inventory_costs(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcula os gastos projetados, custos de overstock (excesso) e stockout (ruptura).

        Args:
            y_true (pd.Series): Faturamento real observado na loja.
            y_pred (np.ndarray): Faturamento previsto pelo modelo preditivo.

        Returns:
            Dict[str, float]: Dicionário contendo os indicadores financeiros consolidados.
        """
        # Converte as predições de faturamento para gasto real de aquisição de estoque (CMV)
        planned_inventory_spend = np.sum(y_pred) * self.cogs_rate
        actual_inventory_needed = np.sum(y_true) * self.cogs_rate

        # Vetor de erros e custos assimétricos
        error_vector = y_pred - y_true
        
        # Overstock (Previsão > Real): Gasto desnecessário com mercadoria parada
        overstock_spend = np.sum(error_vector[error_vector > 0]) * self.cogs_rate
        
        # Stockout (Previsão < Real): Faturamento perdido por falta de produto na prateleira
        stockout_loss = np.sum(np.abs(error_vector[error_vector < 0])) * (1 - self.cogs_rate)

        # Acurácia de planejamento financeiro
        total_actual_revenue = np.sum(y_true)
        mean_absolute_error_percentage = (
            np.sum(np.abs(error_vector)) / total_actual_revenue
        ) * 100
        financial_accuracy = max(0.0, 100.0 - mean_absolute_error_percentage)

        return {
            "total_actual_revenue": float(total_actual_revenue),
            "total_predicted_revenue": float(np.sum(y_pred)),
            "planned_inventory_spend": float(planned_inventory_spend),
            "actual_inventory_needed": float(actual_inventory_needed),
            "overstock_spend": float(overstock_spend),
            "stockout_loss": float(stockout_loss),
            "financial_accuracy": float(financial_accuracy),
        }

    def log_financial_report(self, metrics: Dict[str, float]) -> None:
        """Emite relatório executivo formatado no terminal de operações."""
        logger.info("==========================================================")
        logger.info("--- RELATÓRIO EXECUTIVO DE PLANEJAMENTO FINANCEIRO ---")
        logger.info("==========================================================")
        logger.info(
            f"Faturamento Real (Amostra de Teste)    : $ {metrics['total_actual_revenue']:,.2f}"
        )
        logger.info(
            f"Faturamento Previsto (Modelo RF)       : $ {metrics['total_predicted_revenue']:,.2f}"
        )
        logger.info("----------------------------------------------------------")
        logger.info(
            f"[ORÇAMENTO RECOMENDADO] Gasto de Estoque : $ {metrics['planned_inventory_spend']:,.2f}"
        )
        logger.info(
            f"[CUSTO DE OVERSTOCK] Capital Imobilizado : $ {metrics['overstock_spend']:,.2f}"
        )
        logger.info(
            f"[PERDA DE RUPTURA] Lucro Não Realizado   : $ {metrics['stockout_loss']:,.2f}"
        )
        logger.info("----------------------------------------------------------")
        logger.info(
            f"Acurácia Geral do Planejamento de Gastos : {metrics['financial_accuracy']:.2f}%"
        )
        logger.info("==========================================================")


class SalesForecastingPipeline:
    """Gerencia o fluxo completo de previsão de vendas: leitura, engenharia de features,
    treino do regressor, avaliação de erro e análise financeiro-operacional.
    """

    def __init__(self, dataset_path: str, features: List[str], target: str) -> None:
        self.dataset_path = dataset_path
        self.features = features
        self.target = target
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.financial_analyzer = FinancialImpactAnalyzer(cogs_rate=COGS_RATE)

    def load_data(self) -> pd.DataFrame:
        """Lê o arquivo CSV e garante sua existência antes do processamento.

        Returns:
            pd.DataFrame: Dados brutos carregados do armazenamento local.

        Raises:
            FileNotFoundError: Se o caminho especificado não for localizado.
            RuntimeError: Em caso de falhas no parser de I/O do Pandas.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {self.dataset_path}. "
                "Certifique-se de que o CSV está presente no diretório 'data/'."
            )

        try:
            logger.info(f"Lendo dataset: {self.dataset_path}")
            df = pd.read_csv(self.dataset_path)
            logger.info(f"Dataset carregado com sucesso. Registros: {len(df)}")
            return df
        except Exception as e:
            logger.error("Falha crítica durante a leitura de disco do arquivo CSV.")
            raise RuntimeError(f"Erro no processamento de I/O: {e}") from e

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa transformações de data, extração temporal e validação de esquema.

        Args:
            df (pd.DataFrame): DataFrame bruto.

        Returns:
            pd.DataFrame: DataFrame tratado sem valores nulos na série de tempo.

        Raises:
            KeyError: Se alguma coluna obrigatória estiver ausente da estrutura de dados.
        """
        logger.info("Executando engenharia de recursos e limpeza temporal...")
        df_processed = df.copy()

        # Força parsing explícito no formato europeu/brasileiro original do Walmart (DD-MM-YYYY)
        df_processed["Date"] = pd.to_datetime(
            df_processed["Date"], format="%d-%m-%Y", errors="coerce"
        )

        initial_rows = len(df_processed)
        df_processed = df_processed.dropna(subset=["Date"])
        dropped_rows = initial_rows - len(df_processed)

        if dropped_rows > 0:
            logger.warning(f"Ignorados {dropped_rows} registros por inconsistência de data.")

        df_processed["Year"] = df_processed["Date"].dt.year.astype(int)
        df_processed["Month"] = df_processed["Date"].dt.month.astype(int)
        df_processed["Week"] = (
            df_processed["Date"].dt.isocalendar().week.fillna(0).astype(int)
        )

        expected_columns = set(self.features + [self.target])
        missing_columns = expected_columns - set(df_processed.columns)
        if missing_columns:
            raise KeyError(
                f"Esquema inválido. Colunas obrigatórias ausentes: {missing_columns}"
            )

        return df_processed

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide os dados em variáveis preditoras (X) e variável alvo (y) de treino e teste.

        Args:
            df (pd.DataFrame): Dados processados.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
        """
        train_percentage = int((1 - TEST_SIZE_RATIO) * 100)
        test_percentage = int(TEST_SIZE_RATIO * 100)
        logger.info(
            f"Particionando dataset: {train_percentage}% Treino | {test_percentage}% Teste..."
        )

        X = df[self.features]
        y = df[self.target]

        return train_test_split(
            X, y, test_size=TEST_SIZE_RATIO, random_state=RANDOM_SEED
        )

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Tuple[float, np.ndarray]:
        """Ajusta o algoritmo aos dados e avalia métricas estatísticas e financeiras.

        Args:
            X_train (pd.DataFrame): Features de treino.
            X_test (pd.DataFrame): Features de teste.
            y_train (pd.Series): Target de treino.
            y_test (pd.Series): Target de teste.

        Returns:
            Tuple[float, np.ndarray]: RMSE calculado e o vetor de predições.
        """
        logger.info("Treinando RandomForestRegressor nos núcleos processacionais...")
        self.model.fit(X_train, y_train)
        logger.info("Treinamento concluído com sucesso.")

        logger.info("Executando predições e avaliando erro quadrático...")
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        return float(rmse), predictions

    def run(self) -> None:
        """Orquestra a execução ponta a ponta do pipeline preditivo e analítico."""
        try:
            df = self.load_data()
            df_processed = self.engineer_features(df)
            X_train, X_test, y_train, y_test = self.split_data(df_processed)

            rmse, predictions = self.train_and_evaluate(
                X_train, X_test, y_train, y_test
            )

            logger.info("--- Precisão Estatística do Modelo ---")
            logger.info(f"RMSE (Raiz do Erro Quadrático Médio): $ {rmse:,.2f}")

            # Aciona o módulo financeiro para avaliar os gastos e custos operacionais
            financial_metrics = self.financial_analyzer.evaluate_inventory_costs(
                y_true=y_test, y_pred=predictions
            )
            self.financial_analyzer.log_financial_report(financial_metrics)

        except Exception as e:
            logger.critical(
                f"Execução do pipeline abortada por exceção fatal: {e}",
                exc_info=True,
            )
            raise


def main() -> None:
    """Instancia a fachada principal da aplicação e inicia o fluxo de processamento."""
    pipeline = SalesForecastingPipeline(
        dataset_path=DEFAULT_DATASET_PATH,
        features=FEATURE_COLUMNS,
        target=TARGET_COLUMN,
    )
    pipeline.run()


if __name__ == "__main__":
    main()