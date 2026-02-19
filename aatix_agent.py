"""
Zero-Error Data Analytics AI Agent
===================================
A robust, production-ready autonomous data analysis system.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import sys
from datetime import datetime
import warnings
import traceback

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')
plt.rcParams['figure.max_open_warning'] = 0

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration with safe defaults"""
    auto_clean: bool = True
    generate_viz: bool = True
    max_viz_cols: int = 6
    corr_threshold: float = 0.8
    outlier_zscore: float = 3.0
    output_dir: str = "./analytics_output"
    target_col: Optional[str] = None
    sample_size: int = 10000  # For large datasets
    
    def __post_init__(self):
        """Validate configuration"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.corr_threshold > 1 or self.corr_threshold < 0:
            self.corr_threshold = 0.8


class SafeDataAnalyticsAgent:
    """
    Bulletproof Data Analytics Agent with comprehensive error handling.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.report: Dict[str, Any] = {}
        self.figures: List[str] = []  # Store paths instead of objects
        self.errors: List[str] = []
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("🤖 Agent initialized successfully")
    
    def _safe_execute(self, func, error_msg: str, default_return=None):
        """Execute function with error handling"""
        try:
            return func()
        except Exception as e:
            self.errors.append(f"{error_msg}: {str(e)}")
            logger.error(f"{error_msg}: {str(e)}")
            return default_return
    
    def load_data(self, source: Union[str, Path, pd.DataFrame]) -> 'SafeDataAnalyticsAgent':
        """Load data with multiple format support"""
        logger.info(f"📥 Loading data...")
        
        try:
            if isinstance(source, pd.DataFrame):
                self.df = source.copy()
                source_type = "DataFrame"
            else:
                source = Path(source)
                if not source.exists():
                    raise FileNotFoundError(f"File not found: {source}")
                
                suffix = source.suffix.lower()
                
                if suffix == '.csv':
                    self.df = pd.read_csv(source)
                elif suffix in ['.xlsx', '.xls']:
                    self.df = pd.read_excel(source, engine='openpyxl')
                elif suffix == '.json':
                    self.df = pd.read_json(source)
                elif suffix == '.parquet':
                    self.df = pd.read_parquet(source)
                else:
                    raise ValueError(f"Unsupported format: {suffix}")
                
                source_type = suffix
            
            # Handle large datasets
            if len(self.df) > self.config.sample_size:
                logger.info(f"Large dataset detected. Sampling {self.config.sample_size} rows")
                self.df = self.df.sample(n=self.config.sample_size, random_state=42)
            
            self.original_df = self.df.copy()
            
            self.report['load_info'] = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': list(self.df.columns),
                'dtypes': {k: str(v) for k, v in self.df.dtypes.items()},
                'source_type': source_type,
                'memory_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            
            logger.info(f"✅ Loaded: {len(self.df)} rows × {len(self.df.columns)} columns")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise
        
        return self
    
    def auto_clean(self) -> 'SafeDataAnalyticsAgent':
        """Intelligent data cleaning with safety checks"""
        if not self.config.auto_clean or self.df is None:
            return self
        
        logger.info("🧹 Cleaning data...")
        cleaning_log = {}
        
        try:
            # 1. Remove duplicates
            initial_rows = len(self.df)
            self.df = self.df.drop_duplicates()
            cleaning_log['duplicates_removed'] = initial_rows - len(self.df)
            
            # 2. Handle missing values per column
            for col in self.df.columns:
                missing_pct = self.df[col].isnull().mean()
                
                if missing_pct > 0:
                    if missing_pct > 0.8:  # Drop if >80% missing
                        self.df = self.df.drop(columns=[col])
                        logger.info(f"  Dropped '{col}' ({missing_pct:.1%} missing)")
                    elif pd.api.types.is_numeric_dtype(self.df[col]):
                        # Fill numeric with median
                        median_val = self.df[col].median()
                        self.df[col] = self.df[col].fillna(median_val)
                    else:
                        # Fill categorical with mode (safely)
                        mode_val = self.df[col].mode()
                        if len(mode_val) > 0:
                            self.df[col] = self.df[col].fillna(mode_val[0])
                        else:
                            self.df[col] = self.df[col].fillna('Unknown')
            
            cleaning_log['missing_handled'] = self.original_df.isnull().sum().sum() - self.df.isnull().sum().sum()
            
            # 3. Handle outliers (winsorization)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            
            for col in numeric_cols:
                if col == self.config.target_col:
                    continue
                
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                
                if std_val > 0:  # Avoid division by zero
                    z_scores = np.abs((self.df[col] - mean_val) / std_val)
                    outliers = z_scores > self.config.outlier_zscore
                    
                    if outliers.sum() > 0:
                        # Cap at 1st and 99th percentile
                        low = self.df[col].quantile(0.01)
                        high = self.df[col].quantile(0.99)
                        self.df[col] = self.df[col].clip(low, high)
                        outlier_count += outliers.sum()
            
            cleaning_log['outliers_capped'] = outlier_count
            
            # 4. Optimize types
            for col in self.df.select_dtypes(include=['object']).columns:
                num_unique = self.df[col].nunique()
                num_total = len(self.df)
                
                if num_unique / num_total < 0.5 and num_unique < 100:
                    self.df[col] = self.df[col].astype('category')
            
            self.report['cleaning'] = cleaning_log
            logger.info("✅ Cleaning completed")
            
        except Exception as e:
            logger.error(f"⚠️ Cleaning error (non-critical): {str(e)}")
            self.report['cleaning'] = {'error': str(e), 'status': 'partial'}
        
        return self
    
    def analyze(self) -> 'SafeDataAnalyticsAgent':
        """Comprehensive analysis with error isolation"""
        if self.df is None:
            logger.error("No data loaded")
            return self
        
        logger.info("🔍 Analyzing...")
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'overview': self._get_overview(),
            'statistics': self._get_statistics(),
            'correlations': self._get_correlations(),
            'quality': self._get_quality_metrics(),
            'insights': self._generate_insights()
        }
        
        self.report['analysis'] = analysis_results
        logger.info("✅ Analysis completed")
        return self
    
    def _get_overview(self) -> Dict:
        """Basic dataset overview"""
        return self._safe_execute(
            lambda: {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
                'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)
            },
            "Overview generation failed",
            {}
        )
    
    def _get_statistics(self) -> Dict:
        """Descriptive statistics"""
        stats = {'numeric': {}, 'categorical': {}}
        
        try:
            # Numeric stats
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                desc = self.df[numeric_cols].describe()
                stats['numeric'] = desc.to_dict()
        except Exception as e:
            stats['numeric_error'] = str(e)
        
        try:
            # Categorical stats
            cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols[:5]:  # Limit to first 5
                stats['categorical'][col] = {
                    'unique': int(self.df[col].nunique()),
                    'top_3': self.df[col].value_counts().head(3).to_dict()
                }
        except Exception as e:
            stats['categorical_error'] = str(e)
        
        return stats
    
    def _get_correlations(self) -> Dict:
        """Correlation analysis"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {'message': 'Not enough numeric columns'}
            
            corr_matrix = self.df[numeric_cols].corr().round(3)
            
            # Find high correlations
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > self.config.corr_threshold:
                        high_corr.append({
                            'var1': str(corr_matrix.columns[i]),
                            'var2': str(corr_matrix.columns[j]),
                            'correlation': float(val)
                        })
            
            return {
                'high_correlations': high_corr,
                'count': len(high_corr)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_quality_metrics(self) -> Dict:
        """Data quality assessment"""
        return {
            'missing_values_total': int(self.df.isnull().sum().sum()),
            'missing_by_column': self.df.isnull().sum().to_dict(),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'completeness_pct': round((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100, 2)
        }
    
    def _generate_insights(self) -> List[Dict]:
        """Auto-generate business insights"""
        insights = []
        
        try:
            # 1. Dataset size insight
            insights.append({
                'type': 'scale',
                'message': f"Dataset contains {len(self.df):,} records with {len(self.df.columns)} features",
                'priority': 'info'
            })
            
            # 2. Target variable insight
            if self.config.target_col and self.config.target_col in self.df.columns:
                target = self.df[self.config.target_col]
                if pd.api.types.is_numeric_dtype(target):
                    insights.append({
                        'type': 'target',
                        'message': f"Target '{self.config.target_col}': mean={target.mean():.2f}, std={target.std():.2f}",
                        'priority': 'high'
                    })
            
            # 3. Missing data alert
            missing_total = self.df.isnull().sum().sum()
            if missing_total > 0:
                insights.append({
                    'type': 'quality',
                    'message': f"Found {missing_total} missing values in dataset",
                    'priority': 'medium',
                    'recommendation': 'Review imputation strategy'
                })
            
            # 4. Correlation insight
            if 'analysis' in self.report and 'correlations' in self.report['analysis']:
                high_corr = self.report['analysis']['correlations'].get('high_correlations', [])
                if high_corr:
                    top = high_corr[0]
                    insights.append({
                        'type': 'correlation',
                        'message': f"Strong correlation: {top['var1']} ↔ {top['var2']} (r={top['correlation']:.2f})",
                        'priority': 'medium'
                    })
            
            # 5. Distribution insights
            for col in self.df.select_dtypes(include=[np.number]).columns[:3]:
                skew = self.df[col].skew()
                if abs(skew) > 1:
                    insights.append({
                        'type': 'distribution',
                        'message': f"'{col}' is highly skewed (skewness: {skew:.2f})",
                        'priority': 'low'
                    })
        
        except Exception as e:
            insights.append({'type': 'error', 'message': f'Insight generation error: {str(e)}'})
        
        return insights
    
    def visualize(self) -> 'SafeDataAnalyticsAgent':
        """Generate visualizations with error recovery"""
        if not self.config.generate_viz or self.df is None:
            return self
        
        logger.info("📊 Creating visualizations...")
        
        # Clear any existing figures
        plt.close('all')
        
        # 1. Correlation heatmap
        self._create_heatmap()
        
        # 2. Distribution plots
        self._create_distributions()
        
        # 3. Missing values plot
        if self.original_df.isnull().sum().sum() > 0:
            self._create_missing_plot()
        
        logger.info(f"✅ Created {len(self.figures)} visualizations")
        return self
    
    def _create_heatmap(self):
        """Create correlation heatmap"""
        try:
            numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            
            if len(numeric_cols) < 2:
                return
            
            # Limit columns for readability
            if len(numeric_cols) > 10:
                numeric_cols = numeric_cols[:10]
            
            plt.figure(figsize=(10, 8))
            corr = self.df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, square=True,
                       linewidths=0.5, cbar_kws={"shrink": 0.8})
            
            plt.title('Correlation Matrix', fontsize=14, pad=20)
            plt.tight_layout()
            
            path = f"{self.config.output_dir}/correlation_heatmap.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures.append(path)
            logger.info(f"  ✓ Saved heatmap")
            
        except Exception as e:
            logger.error(f"  ✗ Heatmap failed: {str(e)}")
            plt.close()
    
    def _create_distributions(self):
        """Create distribution plots"""
        try:
            numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
            
            if len(numeric_cols) == 0:
                return
            
            # Select columns (max 6)
            cols_to_plot = numeric_cols[:min(6, len(numeric_cols))]
            n_cols = min(3, len(cols_to_plot))
            n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.reshape(n_rows, n_cols)
            
            for idx, col in enumerate(cols_to_plot):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                # Histogram with KDE
                sns.histplot(self.df[col], kde=True, ax=ax, color='steelblue', alpha=0.7)
                ax.set_title(col, fontsize=10)
                ax.set_xlabel('')
                
                # Add mean line
                mean_val = self.df[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.1f}')
                ax.legend(fontsize=7)
            
            # Hide empty subplots
            for idx in range(len(cols_to_plot), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].set_visible(False)
            
            plt.suptitle('Feature Distributions', fontsize=14, y=1.02)
            plt.tight_layout()
            
            path = f"{self.config.output_dir}/distributions.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures.append(path)
            logger.info(f"  ✓ Saved distributions")
            
        except Exception as e:
            logger.error(f"  ✗ Distributions failed: {str(e)}")
            plt.close()
    
    def _create_missing_plot(self):
        """Create missing values visualization"""
        try:
            if self.original_df.isnull().sum().sum() == 0:
                return
            
            plt.figure(figsize=(12, 6))
            
            # Calculate missing percentage by column
            missing_pct = (self.original_df.isnull().sum() / len(self.original_df) * 100).sort_values(ascending=False)
            missing_pct = missing_pct[missing_pct > 0]
            
            if len(missing_pct) == 0:
                return
            
            # Bar plot
            ax = missing_pct.plot(kind='bar', color='coral')
            plt.title('Missing Values by Column (%)', fontsize=14)
            plt.xlabel('Columns')
            plt.ylabel('Missing %')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for i, v in enumerate(missing_pct):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=8)
            
            plt.tight_layout()
            
            path = f"{self.config.output_dir}/missing_values.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.figures.append(path)
            logger.info(f"  ✓ Saved missing values plot")
            
        except Exception as e:
            logger.error(f"  ✗ Missing values plot failed: {str(e)}")
            plt.close()
    
    def generate_report(self, format: str = 'json') -> str:
        """Generate and save report"""
        logger.info(f"📝 Generating {format} report...")
        
        try:
            if format == 'json':
                # Clean report for JSON serialization
                clean_report = self._clean_for_json(self.report)
                
                path = f"{self.config.output_dir}/analysis_report.json"
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(clean_report, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✅ JSON report saved: {path}")
                return path
            
            elif format == 'markdown':
                md_content = self._generate_markdown()
                path = f"{self.config.output_dir}/analysis_report.md"
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                
                logger.info(f"✅ Markdown report saved: {path}")
                return path
            
            else:
                raise ValueError(f"Unknown format: {format}")
        
        except Exception as e:
            logger.error(f"❌ Report generation failed: {str(e)}")
            return ""
    
    def _clean_for_json(self, obj):
        """Convert objects to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def _generate_markdown(self) -> str:
        """Generate markdown report"""
        lines = [
            "# Data Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Overview",
            f"- **Rows**: {self.report.get('load_info', {}).get('rows', 'N/A'):,}",
            f"- **Columns**: {self.report.get('load_info', {}).get('columns', 'N/A')}",
            f"- **Memory**: {self.report.get('load_info', {}).get('memory_mb', 'N/A')} MB",
            "",
            "## Data Quality",
        ]
        
        # Quality metrics
        quality = self.report.get('analysis', {}).get('quality', {})
        lines.extend([
            f"- **Completeness**: {quality.get('completeness_pct', 'N/A')}%",
            f"- **Duplicate Rows**: {quality.get('duplicate_rows', 'N/A')}",
            f"- **Total Missing**: {quality.get('missing_values_total', 'N/A')}",
            "",
            "## Key Insights",
        ])
        
        # Insights
        insights = self.report.get('analysis', {}).get('insights', [])
        for insight in insights:
            lines.append(f"### {insight.get('type', 'Insight').title()}")
            lines.append(f"- {insight.get('message', '')}")
            if 'recommendation' in insight:
                lines.append(f"- **Recommendation**: {insight['recommendation']}")
            lines.append("")
        
        # Cleaning summary
        if 'cleaning' in self.report:
            lines.extend([
                "## Data Cleaning",
                f"- Duplicates removed: {self.report['cleaning'].get('duplicates_removed', 0)}",
                f"- Missing values handled: {self.report['cleaning'].get('missing_handled', 0)}",
                f"- Outliers capped: {self.report['cleaning'].get('outliers_capped', 0)}",
                "",
            ])
        
        # Errors (if any)
        if self.errors:
            lines.extend([
                "## Errors Encountered",
                "The following non-critical errors occurred:",
            ])
            for error in self.errors:
                lines.append(f"- {error}")
        
        lines.extend([
            "",
            "## Visualizations",
            "The following charts were generated:",
            f"1. Correlation Heatmap: `correlation_heatmap.png`",
            f"2. Distributions: `distributions.png`",
            f"3. Missing Values: `missing_values.png` (if applicable)",
        ])
        
        return "\n".join(lines)
    
    def run_pipeline(self, source: Union[str, Path, pd.DataFrame]) -> Dict[str, Any]:
        """Execute complete analysis pipeline"""
        logger.info("🚀 Starting full pipeline...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load
            self.load_data(source)
            
            # Step 2: Clean
            self.auto_clean()
            
            # Step 3: Analyze
            self.analyze()
            
            # Step 4: Visualize
            self.visualize()
            
            # Step 5: Report
            json_path = self.generate_report('json')
            md_path = self.generate_report('markdown')
            
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'duration_seconds': round(duration, 2),
                'output_directory': self.config.output_dir,
                'files_generated': {
                    'json_report': json_path,
                    'markdown_report': md_path,
                    'visualizations': self.figures
                },
                'data_shape': {
                    'original_rows': len(self.original_df) if self.original_df is not None else 0,
                    'final_rows': len(self.df) if self.df is not None else 0,
                    'columns': len(self.df.columns) if self.df is not None else 0
                },
                'errors': self.errors if self.errors else None
            }
            
            logger.info("✅ Pipeline completed successfully!")
            logger.info(f"⏱️  Duration: {duration:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def demo_with_sample_data():
    """Run demonstration with automatically generated data"""
    print("=" * 60)
    print("RUNNING DEMO WITH SAMPLE DATA")
    print("=" * 60)
    
    # Create realistic sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 12, n_samples).clip(18, 80).astype(int),
        'income': np.random.lognormal(10.8, 0.5, n_samples).astype(int),
        'spending_score': np.random.beta(2, 5, n_samples) * 100,
        'membership_years': np.random.poisson(3, n_samples),
        'purchase_frequency': np.random.gamma(2, 2, n_samples),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.35]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'churned': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    })
    
    # Introduce some realistic issues
    # Missing values
    missing_idx = np.random.choice(n_samples, 50, replace=False)
    data.loc[missing_idx[:25], 'age'] = np.nan
    data.loc[missing_idx[25:], 'income'] = np.nan
    
    # Outliers
    data.loc[0, 'income'] = 5000000  # Extreme outlier
    
    # Duplicates
    data = pd.concat([data, data.iloc[:5]], ignore_index=True)
    
    print(f"\n📊 Created sample dataset: {len(data)} rows")
    print(f"   Issues injected: 25 missing ages, 25 missing incomes, 1 outlier, 5 duplicates")
    
    # Run agent
    config = AgentConfig(
        target_col='churned',
        output_dir='./demo_output',
        auto_clean=True,
        generate_viz=True
    )
    
    agent = SafeDataAnalyticsAgent(config)
    result = agent.run_pipeline(data)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if result['status'] == 'success':
        print(f"✅ Status: Success")
        print(f"⏱️  Duration: {result['duration_seconds']}s")
        print(f"📁 Output: {result['output_directory']}")
        print(f"📊 Files generated: {len(result['files_generated']['visualizations']) + 2}")
        
        if result['errors']:
            print(f"\n⚠️  Non-critical errors: {len(result['errors'])}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    return result


def analyze_csv_file(filepath: str):
    """Analyze an existing CSV file"""
    print(f"\nAnalyzing file: {filepath}")
    
    config = AgentConfig(
        output_dir='./analysis_results',
        auto_clean=True,
        generate_viz=True
    )
    
    agent = SafeDataAnalyticsAgent(config)
    result = agent.run_pipeline(filepath)
    
    if result['status'] == 'success':
        print(f"\n✅ Analysis complete! Check: {result['output_directory']}")
    else:
        print(f"\n❌ Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    analyze_csv_file("Amazon-Sales-Report-40K-Rows.csv")


    
  