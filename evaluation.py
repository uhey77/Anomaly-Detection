# evaluation.py - 異常検知評価モジュール
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

class AnomalyEvaluator:
    """異常検知アルゴリズムの評価を行うクラス"""
    
    def __init__(self, true_anomaly_dates=None):
        """
        評価器を初期化
        
        Args:
            true_anomaly_dates (list, optional): 既知の異常日付のリスト。
                                              例: ["1987-10-19", "2008-10-13", "2020-03-16"]
        """
        self.true_anomaly_dates = true_anomaly_dates or []
    
    def evaluate(self, df, detected_anomalies, delay_tolerance=0):
        """
        異常検知の評価を実行
        
        Args:
            df (pd.DataFrame): 元の時系列データ
            detected_anomalies (pd.DataFrame): 検出された異常データ
            delay_tolerance (int): 検知遅延の許容値（日数）
            
        Returns:
            dict: 評価指標
        """
        # データフレームのチェック
        if df.empty or 'Date' not in df.columns:
            raise ValueError("入力データフレームが無効です")
        
        # 検出された異常の日付を取得
        if detected_anomalies.empty:
            detected_dates = set()
        else:
            detected_dates = set(detected_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist())
        
        # 既知の異常日付
        known_dates = set(self.true_anomaly_dates)
        
        # 許容遅延を適用した検出セットを作成
        if delay_tolerance > 0 and not detected_anomalies.empty:
            # 日付を日時インデックスに変換
            df_dates = pd.DatetimeIndex(df['Date'])
            
            # 拡張された検出日の集合を初期化
            expanded_detected_dates = set()
            
            # 検出された各日付に対して
            for date_str in detected_dates:
                date = pd.to_datetime(date_str)
                expanded_detected_dates.add(date_str)
                
                # 許容遅延内に含まれる日付も追加
                for i in range(1, delay_tolerance + 1):
                    try:
                        future_date = df_dates[df_dates.get_loc(date) + i]
                        expanded_detected_dates.add(future_date.strftime('%Y-%m-%d'))
                    except:
                        pass  # インデックス範囲外
            
            detected_dates = expanded_detected_dates
        
        # 評価指標の計算
        true_positives = len(detected_dates.intersection(known_dates))
        false_positives = len(detected_dates - known_dates)
        false_negatives = len(known_dates - detected_dates)
        
        # 全日付を取得して真陰性を計算
        all_dates = set(df['Date'].dt.strftime('%Y-%m-%d').tolist())
        true_negatives = len(all_dates - detected_dates - known_dates)
        
        # 指標計算
        total = true_positives + false_positives + false_negatives + true_negatives
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # F-beta スコア（beta=2 は再現率を重視）
        beta = 2
        f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) > 0 else 0
        
        # 検知遅延の計算（該当する場合）
        detection_delay = self._calculate_detection_delay(df, detected_anomalies, known_dates)
        
        # 評価結果を辞書として返す
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'f_beta_score': f_beta,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'detection_delay': detection_delay
        }
    
    def _calculate_detection_delay(self, df, detected_anomalies, known_dates):
        """
        検知遅延を計算
        
        Args:
            df (pd.DataFrame): 元の時系列データ
            detected_anomalies (pd.DataFrame): 検出された異常データ
            known_dates (set): 既知の異常日付の集合
            
        Returns:
            float: 平均検知遅延（検出された異常がない場合はNone）
        """
        if detected_anomalies.empty or not known_dates:
            return None
        
        df_dates = pd.DatetimeIndex(df['Date'])
        delays = []
        
        # 検出された各異常に対して最も近い真の異常を見つける
        for idx, anomaly in detected_anomalies.iterrows():
            anomaly_date = anomaly['Date']
            anomaly_date_str = anomaly_date.strftime('%Y-%m-%d')
            
            # すでに既知の異常日ならば遅延はゼロ
            if anomaly_date_str in known_dates:
                delays.append(0)
                continue
            
            # 最も近い既知の異常日を見つける
            min_delay = float('inf')
            for known_date_str in known_dates:
                known_date = pd.to_datetime(known_date_str)
                
                # 検出日と真の異常日の差（日数）を計算
                try:
                    known_idx = df_dates.get_loc(known_date)
                    detected_idx = df_dates.get_loc(anomaly_date)
                    delay = detected_idx - known_idx
                    
                    # 正の遅延のみを考慮（検出が異常の後）
                    if delay > 0:
                        min_delay = min(min_delay, delay)
                except:
                    continue
            
            # 有効な遅延が見つかった場合のみ追加
            if min_delay != float('inf'):
                delays.append(min_delay)
        
        # 平均遅延を計算
        return np.mean(delays) if delays else None
    
    def plot_roc_curve(self, y_true, y_score):
        """
        ROC曲線をプロット
        
        Args:
            y_true (array): 真のラベル（0=正常、1=異常）
            y_score (array): 異常スコア
            
        Returns:
            plotly.graph_objects.Figure: ROC曲線のプロット
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC曲線 (AUC = {roc_auc:.3f})'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='ランダム予測'
        ))
        
        fig.update_layout(
            title='ROC曲線',
            xaxis_title='偽陽性率 (False Positive Rate)',
            yaxis_title='真陽性率 (True Positive Rate)',
            template='plotly_white',
            height=500,
            width=700
        )
        
        return fig
    
    def plot_pr_curve(self, y_true, y_score):
        """
        Precision-Recall曲線をプロット
        
        Args:
            y_true (array): 真のラベル（0=正常、1=異常）
            y_score (array): 異常スコア
            
        Returns:
            plotly.graph_objects.Figure: PR曲線のプロット
        """
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR曲線 (AUC = {pr_auc:.3f})'
        ))
        
        # ランダム予測のベースライン（異常比率）
        baseline = sum(y_true) / len(y_true)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[baseline, baseline],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name=f'ベースライン (異常比率 = {baseline:.3f})'
        ))
        
        fig.update_layout(
            title='適合率-再現率曲線',
            xaxis_title='再現率 (Recall)',
            yaxis_title='適合率 (Precision)',
            template='plotly_white',
            height=500,
            width=700
        )
        
        return fig
    
    def compare_methods(self, df, methods, thresholds, known_anomalies=None):
        """
        複数の異常検知手法を比較
        
        Args:
            df (pd.DataFrame): 時系列データ
            methods (list): 検出手法のリスト（例：['z_score', 'iqr', 'moving_avg']）
            thresholds (list): 閾値のリスト（例：[2.0, 2.5, 3.0, 3.5]）
            known_anomalies (list, optional): 既知の異常日付のリスト
            
        Returns:
            pd.DataFrame: 比較結果
        """
        from detection.anomaly_detector import AnomalyDetector
        
        # 既知の異常があれば設定
        if known_anomalies:
            self.true_anomaly_dates = known_anomalies
        
        results = []
        
        for method in methods:
            for threshold in thresholds:
                # 異常検知器を作成
                detector = AnomalyDetector(method=method, threshold=threshold)
                
                # 異常検知を実行
                detected_anomalies = detector.detect(df)
                
                # 評価を実行
                eval_result = self.evaluate(df, detected_anomalies)
                
                # 結果を格納
                results.append({
                    '検出手法': method,
                    '閾値': threshold,
                    '適合率': eval_result['precision'],
                    '再現率': eval_result['recall'],
                    'F1スコア': eval_result['f1_score'],
                    'Fβスコア': eval_result['f_beta_score'],
                    '検知数': len(detected_anomalies),
                    '真陽性': eval_result['true_positives'],
                    '偽陽性': eval_result['false_positives'],
                    '偽陰性': eval_result['false_negatives']
                })
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        
        return results_df
