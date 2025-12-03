"""
AI Chatbot Module for CortexX Forecasting Platform
Professional Rule-Based Assistant - No External APIs Required
"""

import streamlit as st
from typing import List, Dict, Tuple
from datetime import datetime


class CortexXChatbot:
    """
    Rule-based AI chatbot for CortexX Forecasting Platform.
    Provides context-aware assistance for ML models, features, and navigation.
    """

    def __init__(self):
        # 9 models in your project
        self.models = [
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "Random Forest",
            "Lasso",
            "Ridge",
            "Linear Regression",
            "Decision Tree",
            "KNN",
        ]

        # Pages in your app
        self.pages = {
            "Dashboard": "1_🏠_Dashboard.py",
            "Data Exploration": "2_📊_Data_Exploration.py",
            "Feature Engineering": "3_⚙️_Feature_Engineering.py",
            "Model Training": "4_🤖_Model_Training.py",
            "Forecasting": "5_📈_Forecasting.py",
            "Model Evaluation": "6_📋_Model_Evaluation.py",
        }

        self.knowledge_base = self._build_knowledge_base()

    def _build_knowledge_base(self) -> Dict[str, Dict]:
        """Build simple rule-based knowledge base."""
        return {
            "xgboost": {
                "keywords": ["xgboost", "xgb"],
                "response": (
                    "**XGBoost** is a gradient boosting algorithm that works well on "
                    "tabular forecasting problems and supports many hyperparameters "
                    "such as learning_rate, max_depth, n_estimators, subsample and colsample_bytree."
                ),
                "suggestions": ["Explain LightGBM", "Compare models", "How to tune XGBoost?"],
            },
            "lightgbm": {
                "keywords": ["lightgbm", "lgbm"],
                "response": (
                    "**LightGBM** is a fast, memory‑efficient gradient boosting library. "
                    "It is well suited for large datasets and supports leaf‑wise tree growth."
                ),
                "suggestions": ["Explain XGBoost", "Training speed tips", "When to use LightGBM?"],
            },
            "catboost": {
                "keywords": ["catboost"],
                "response": (
                    "**CatBoost** handles categorical features internally and is robust "
                    "to overfitting with good default hyperparameters."
                ),
                "suggestions": ["Categorical features", "Compare with LightGBM", "Hyperparameters"],
            },
            "random_forest": {
                "keywords": ["random forest", "rf"],
                "response": (
                    "**Random Forest** is an ensemble of decision trees that reduces variance "
                    "and handles non‑linear relationships with parameters like n_estimators and max_depth."
                ),
                "suggestions": ["Feature importance", "Avoid overfitting", "When to use RF?"],
            },
            "lasso": {
                "keywords": ["lasso"],
                "response": (
                    "**Lasso Regression** uses L1 regularization to perform feature selection by "
                    "shrinking some coefficients exactly to zero; key parameter is alpha."
                ),
                "suggestions": ["Lasso vs Ridge", "Feature selection", "Regularization strength"],
            },
            "ridge": {
                "keywords": ["ridge"],
                "response": (
                    "**Ridge Regression** uses L2 regularization to control coefficient size and "
                    "reduce overfitting while keeping all features; key parameter is alpha."
                ),
                "suggestions": ["Ridge vs Lasso", "Regularization", "When to use Ridge?"],
            },
            "linear": {
                "keywords": ["linear regression", "linear model", "ols"],
                "response": (
                    "**Linear Regression** is a simple baseline model that assumes a linear "
                    "relationship between features and target and is easy to interpret."
                ),
                "suggestions": ["Assumptions", "When linear fails?", "Alternatives"],
            },
            "decision_tree": {
                "keywords": ["decision tree", "tree", "dt"],
                "response": (
                    "**Decision Trees** split the data into regions using simple rules and are "
                    "very interpretable but can overfit without depth and split constraints."
                ),
                "suggestions": ["Pruning", "Overfitting", "Tree depth"],
            },
            "knn": {
                "keywords": ["knn", "nearest neighbor"],
                "response": (
                    "**K‑Nearest Neighbors** predicts from the closest historical points and is "
                    "sensitive to feature scaling and the number of neighbors k."
                ),
                "suggestions": ["Choosing k", "Scaling features", "Performance tips"],
            },
            "backtesting": {
                "keywords": ["backtest", "walk forward", "validation"],
                "response": (
                    "**Backtesting** (walk‑forward validation) evaluates models on rolling "
                    "time windows and is configured in the **Model Training** page."
                ),
                "suggestions": ["Backtesting steps", "Window size", "Evaluation page"],
            },
            "hyperparameter": {
                "keywords": ["hyperparameter", "tuning", "optimization"],
                "response": (
                    "**Hyperparameter optimization** runs multiple configurations to find the "
                    "best settings; in this project it is handled in the optimization module "
                    "and exposed in the **Model Training** page."
                ),
                "suggestions": ["Which params to tune?", "Time vs accuracy", "XGBoost tuning"],
            },
            "forecasting": {
                "keywords": ["forecast", "prediction", "future"],
                "response": (
                    "**Forecasting** is performed in the **Forecasting** page where you select "
                    "a trained model, forecast horizon and optionally prediction intervals."
                ),
                "suggestions": ["Forecast horizon", "Prediction intervals", "Best model"],
            },
            "feature_engineering": {
                "keywords": ["feature", "lag", "rolling", "engineering"],
                "response": (
                    "**Feature Engineering** creates time‑series features like lags, rolling "
                    "means and trends and is implemented in the **Feature Engineering** page."
                ),
                "suggestions": ["Which lags to use?", "Rolling windows", "Seasonality features"],
            },
            "data_quality": {
                "keywords": ["quality", "missing", "duplicates", "outliers"],
                "response": (
                    "The **Data Quality Dashboard** on the **Dashboard** page summarizes "
                    "missing values, duplicates, outliers and freshness metrics."
                ),
                "suggestions": ["Handle missing data", "Remove duplicates", "Outlier strategy"],
            },
            "model_evaluation": {
                "keywords": ["evaluation", "metrics", "performance", "rmse", "mae", "mape"],
                "response": (
                    "**Model Evaluation** compares models using metrics such as RMSE, MAE, "
                    "MAPE and R² in the **Model Evaluation** page."
                ),
                "suggestions": ["Best metric?", "Compare models", "Export results"],
            },
            "workflow": {
                "keywords": ["workflow", "process", "steps", "how to start"],
                "response": (
                    "A typical **workflow** is:\n"
                    "1. Data Exploration → load and inspect data\n"
                    "2. Feature Engineering → create time‑series features\n"
                    "3. Model Training → train the 9 ML models with backtesting\n"
                    "4. Model Evaluation → compare metrics\n"
                    "5. Forecasting → generate future predictions."
                ),
                "suggestions": ["Data Exploration", "Model Training", "Forecasting"],
            },
        }

    def get_response(self, user_message: str) -> Tuple[str, List[str]]:
        """Return (response_text, suggestions) for a user message."""
        text = (user_message or "").lower().strip()

        if not text:
            return (
                "Please type a question about models, forecasting, or navigation.",
                ["Show all models", "Explain workflow", "Where is Forecasting page?"],
            )

        if any(w in text for w in ["hi", "hello", "hey", "salam"]):
            return self._greeting_response()

        if "help" in text or "what can you do" in text:
            return self._help_response()

        if "models" in text or "algorithms" in text:
            return self._models_list_response()

        for _, data in self.knowledge_base.items():
            if any(k in text for k in data["keywords"]):
                return data["response"], data["suggestions"]

        return self._default_response()

    def _greeting_response(self) -> Tuple[str, List[str]]:
        return (
            "👋 Welcome to **CortexX Forecasting**. Ask about models, backtesting, "
            "feature engineering, evaluation, or how to navigate the pages.",
            ["Show all models", "Explain workflow", "Where is Model Training page?"],
        )

    def _help_response(self) -> Tuple[str, List[str]]:
        return (
            "I can explain the 9 ML models, backtesting, hyperparameter optimization, "
            "data quality, and how to use each page in the platform.",
            ["Show all models", "Explain XGBoost", "Explain workflow"],
        )

    def _models_list_response(self) -> Tuple[str, List[str]]:
        text = (
            "These **9 ML models** are available in CortexX:\n"
            "- XGBoost\n"
            "- LightGBM\n"
            "- CatBoost\n"
            "- Random Forest\n"
            "- Lasso Regression\n"
            "- Ridge Regression\n"
            "- Linear Regression\n"
            "- Decision Tree\n"
            "- KNN\n"
            "Ask about any model for more details."
        )
        return text, ["Explain XGBoost", "Explain LightGBM", "Compare models"]

    def _default_response(self) -> Tuple[str, List[str]]:
        return (
            "I did not understand that question. Try asking about models, "
            "feature engineering, backtesting, forecasting, or the workflow.",
            ["Show all models", "Explain workflow", "Where is Forecasting page?"],
        )

    def export_chat_history(self, messages: List[Dict]) -> str:
        """Export chat history to plain text."""
        header = (
            f"CortexX Chatbot Conversation - "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 70
            + "\n\n"
        )
        lines = [header]
        for msg in messages:
            role = "You" if msg.get("role") == "user" else "CortexX Assistant"
            ts = msg.get("timestamp", "")
            content = msg.get("content", "")
            lines.append(f"[{ts}] {role}:\n{content}\n\n")
        return "".join(lines)


def get_chatbot() -> CortexXChatbot:
    """Factory to create chatbot."""
    return CortexXChatbot()


def render_chatbot_ui():
    """
    Render a simple chatbot UI in the Streamlit sidebar.
    Call this once from the Dashboard page.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🤖 CortexX AI Assistant")

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = get_chatbot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am the CortexX assistant. "
                    "Ask me about models, workflow, or any page in the app."
                ),
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        ]

    # Show last few messages
    for msg in st.session_state.chat_history[-6:]:
        who = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
        st.sidebar.markdown(
            f"**{who} [{msg['timestamp']}]**  \n{msg['content']}"
        )

    user_input = st.sidebar.text_input(
        "Type your question:", key="chat_input", placeholder="e.g. Explain XGBoost"
    )

    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    send_clicked = col1.button("Send")
    clear_clicked = col2.button("Clear")
    export_clicked = col3.button("Export")

    if send_clicked and user_input:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        response, _ = st.session_state.chatbot.get_response(user_input)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        st.experimental_rerun()

    if clear_clicked:
        st.session_state.chat_history = []
        st.experimental_rerun()

    if export_clicked and st.session_state.chat_history:
        export_text = st.session_state.chatbot.export_chat_history(
            st.session_state.chat_history
        )
        st.sidebar.download_button(
            label="Download chat.txt",
            data=export_text,
            file_name="cortexx_chat.txt",
            mime="text/plain",
        )
