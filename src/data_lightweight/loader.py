"""
轻量化数据加载器

从Qlib风格的二进制文件加载轻量化特征数据，供训练使用。
"""

import struct
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


class LightweightDataLoader:
    """
    轻量化数据加载器
    
    从Qlib格式的二进制文件加载数据：
    - 加载交易日历
    - 加载股票列表
    - 加载特征数据（支持单只股票、单个特征）
    - 支持多频率数据加载
    
    Attributes:
        data_dir (Path): 数据目录路径
        calendar (List[str]): 交易日历
        instruments (List[str]): 股票列表
        feature_names (List[str]): 特征名称列表
    """
    
    def __init__(self, data_dir: str = "src/data-lightweight/data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
        self.calendar: List[str] = []
        self.instruments: List[str] = []
        self.feature_names: List[str] = []
        
        self._load_metadata()
        
        logger.info(f"LightweightDataLoader初始化完成")
        logger.info(f"  数据目录: {self.data_dir}")
        logger.info(f"  交易日数: {len(self.calendar)}")
        logger.info(f"  股票数: {len(self.instruments)}")
        logger.info(f"  特征数: {len(self.feature_names)}")
        
    def _load_metadata(self) -> None:
        """加载元数据"""
        cal_file = self.data_dir / "calendars" / "day.txt"
        if cal_file.exists():
            with open(cal_file, 'r') as f:
                self.calendar = [line.strip() for line in f if line.strip()]
                
        inst_file = self.data_dir / "instruments" / "csi300.txt"
        if not inst_file.exists():
            inst_file = self.data_dir / "instruments" / "all.txt"
            
        if inst_file.exists():
            with open(inst_file, 'r') as f:
                self.instruments = [line.strip().split()[0] for line in f if line.strip()]
                
        if self.instruments:
            sample_stock_dir = self.data_dir / "features" / self.instruments[0]
            if sample_stock_dir.exists():
                bin_files = list(sample_stock_dir.glob("*.day.bin"))
                self.feature_names = sorted([f.stem.replace('.day', '') for f in bin_files])
                
    def load_bin_feature(
        self,
        stock_code: str,
        feature_name: str,
        frequency: str = 'day'
    ) -> np.ndarray:
        """
        加载单个特征
        
        Args:
            stock_code: 股票代码
            feature_name: 特征名称
            frequency: 频率类型
            
        Returns:
            特征值数组
        """
        feature_file = self.data_dir / "features" / stock_code / f"{feature_name}.{frequency}.bin"
        
        if not feature_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")
            
        with open(feature_file, 'rb') as f:
            data = f.read()
            
        count = struct.unpack('I', data[:4])[0]
        values = np.array(struct.unpack(f'{count}f', data[4:]), dtype=np.float32)
        
        return values
        
    def load_stock_features(
        self,
        stock_code: str,
        features: Optional[List[str]] = None,
        frequency: str = 'day'
    ) -> pd.DataFrame:
        """
        加载单只股票的所有特征
        
        Args:
            stock_code: 股票代码
            features: 特征列表（None表示加载所有特征）
            frequency: 频率类型
            
        Returns:
            特征DataFrame
        """
        features = features or self.feature_names
        
        data = {}
        for feature_name in features:
            try:
                values = self.load_bin_feature(stock_code, feature_name, frequency)
                data[feature_name] = values
            except FileNotFoundError:
                logger.warning(f"特征 {feature_name} 不存在，跳过")
                
        df = pd.DataFrame(data)
        
        if len(self.calendar) >= len(df):
            df.index = pd.to_datetime(self.calendar[:len(df)])
        else:
            df.index = pd.to_datetime(self.calendar)
            
        df.index.name = 'trade_date'
        
        return df
        
    def load_all_features(
        self,
        features: Optional[List[str]] = None,
        frequency: str = 'day',
        stock_list: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        加载所有股票的特征数据
        
        Args:
            features: 特征列表
            frequency: 频率类型
            stock_list: 股票列表
            
        Returns:
            股票代码到特征DataFrame的映射
        """
        stock_list = stock_list or self.instruments
        features = features or self.feature_names
        
        all_data = {}
        
        for i, stock_code in enumerate(stock_list):
            try:
                df = self.load_stock_features(stock_code, features, frequency)
                all_data[stock_code] = df
                
                if (i + 1) % 50 == 0 or (i + 1) == len(stock_list):
                    logger.info(f"特征加载进度: {i+1}/{len(stock_list)} ({(i+1)/len(stock_list)*100:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"加载 {stock_code} 失败: {e}")
                
        logger.info(f"特征加载完成: {len(all_data)}只股票")
        return all_data
        
    def get_close_prices(
        self,
        stock_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取收盘价数据（用于训练）
        
        Args:
            stock_list: 股票列表
            
        Returns:
            收盘价DataFrame（日期×股票）
        """
        stock_list = stock_list or self.instruments
        
        close_data = {}
        for stock_code in stock_list:
            try:
                if 'close_norm' in self.feature_names:
                    values = self.load_bin_feature(stock_code, 'close_norm', 'day')
                elif 'close' in self.feature_names:
                    values = self.load_bin_feature(stock_code, 'close', 'day')
                else:
                    logger.warning(f"找不到close特征")
                    continue
                    
                close_data[stock_code] = values
            except Exception as e:
                logger.warning(f"加载 {stock_code} close失败: {e}")
                
        df = pd.DataFrame(close_data)
        
        if len(self.calendar) >= len(df):
            df.index = pd.to_datetime(self.calendar[:len(df)])
        else:
            df.index = pd.to_datetime(self.calendar)
            
        df.index.name = 'trade_date'
        
        return df
        
    def get_feature_matrix(
        self,
        feature_name: str,
        stock_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        获取单个特征的所有股票数据
        
        Args:
            feature_name: 特征名称
            stock_list: 股票列表
            
        Returns:
            特征矩阵（日期×股票）
        """
        stock_list = stock_list or self.instruments
        
        feature_data = {}
        for stock_code in stock_list:
            try:
                values = self.load_bin_feature(stock_code, feature_name, 'day')
                feature_data[stock_code] = values
            except Exception as e:
                logger.warning(f"加载 {stock_code} {feature_name}失败: {e}")
                
        df = pd.DataFrame(feature_data)
        
        if len(self.calendar) >= len(df):
            df.index = pd.to_datetime(self.calendar[:len(df)])
        else:
            df.index = pd.to_datetime(self.calendar)
            
        df.index.name = 'trade_date'
        
        return df
        
    def get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        return self.feature_names.copy()
        
    def get_date_range(self) -> Tuple[str, str]:
        """获取数据日期范围"""
        if self.calendar:
            return self.calendar[0], self.calendar[-1]
        return "", ""


if __name__ == "__main__":
    print("=" * 60)
    print("LightweightDataLoader 单元测试")
    print("=" * 60)
    
    loader = LightweightDataLoader("src/data-lightweight/test_data")
    
    print("\n【测试1】元数据加载")
    print(f"交易日数: {len(loader.calendar)}")
    print(f"股票数: {len(loader.instruments)}")
    print(f"特征数: {len(loader.feature_names)}")
    
    if loader.instruments and loader.feature_names:
        print("\n【测试2】加载单个特征")
        stock_code = loader.instruments[0]
        feature_name = loader.feature_names[0]
        
        try:
            values = loader.load_bin_feature(stock_code, feature_name, 'day')
            print(f"股票: {stock_code}, 特征: {feature_name}")
            print(f"数据长度: {len(values)}")
            print(f"前5个值: {values[:5]}")
        except Exception as e:
            print(f"加载失败: {e}")
            
        print("\n【测试3】加载股票特征")
        try:
            df = loader.load_stock_features(stock_code)
            print(f"特征矩阵: {df.shape}")
            print(df.head())
        except Exception as e:
            print(f"加载失败: {e}")
            
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
