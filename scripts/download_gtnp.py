"""
GTN-P 冻土地温数据下载脚本

【重要】GTN-P 数据完全公开，CC BY 4.0 许可，无需登录！
API 文档: https://data.gtn-p.org/api/swagger

用法:
    # 1. 下载中国区域所有冻土温度站点元数据
    python download_gtnp.py --list-china

    # 2. 下载特定站点的地温数据
    python download_gtnp.py --download-pt --site-id 123

    # 3. 批量下载中国区域所有数据
    python download_gtnp.py --download-china

    # 4. 下载 PANGAEA 全球 MAGT 数据集（41年地温）
    python download_gtnp.py --download-pangaea

    # 5. 指定下载目录
    python download_gtnp.py --download-china --download-dir /root/autodl-tmp/gtnp_data
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Optional, List
from datetime import datetime


# ============================================================
# GTN-P API 配置
# ============================================================

GTNP_API_BASE = "https://data.gtn-p.org/api"
GTNP_DATA_PLATFORM = "https://data.gtn-p.org"

# PANGAEA 数据集 DOI
PANGAEA_MAGT_URL = "https://doi.pangaea.de/10.1594/PANGAEA.972992"

# 默认下载目录
DEFAULT_DOWNLOAD_DIR = "/root/autodl-tmp/gtnp_data"


# ============================================================
# GTN-P API 客户端
# ============================================================

class GTNPClient:
    """GTN-P 数据下载客户端（无需登录，公开 API）"""

    def __init__(self, download_dir: str = DEFAULT_DOWNLOAD_DIR):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36",
            "Accept": "application/json",
        })
        self.download_log = []
        self.log_file = self.download_dir / "download_log.json"

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 站点列表 ----

    def list_sites(self, country: str = None, borehole_data: bool = True,
                   metadata: bool = True) -> list:
        """
        获取站点列表

        Args:
            country: 国家名（如 "China"）
            borehole_data: 是否包含有钻孔数据的站点
            metadata: 是否包含元数据

        Returns:
            list: 站点列表
        """
        url = f"{GTNP_API_BASE}/list-sites"
        params = {
            "borehole_data": str(borehole_data).lower(),
            "metadata": str(metadata).lower(),
        }
        if country:
            params["country"] = country

        print(f"[{self._now()}] 📋 查询站点列表: country={country or 'All'}")
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                return [data]
        except Exception as e:
            print(f"[{self._now()}] ❌ 查询失败: {e}")
            return []

    def list_china_sites(self):
        """列出中国区域的冻土温度站点"""
        sites = self.list_sites(country="China")
        print(f"\n{'='*60}")
        print(f"🇨🇳 中国冻土温度站点列表")
        print(f"{'='*60}")
        if not sites:
            print("  ⚠️  未找到站点，可能 API 格式有变化")
            print("  💡 建议：直接访问 https://data.gtn-p.org/ 在线查看")
            return sites

        for i, site in enumerate(sites[:50], 1):  # 最多显示50个
            name = site.get("name", site.get("site_name", "未知"))
            lat = site.get("latitude", site.get("lat", "?"))
            lon = site.get("longitude", site.get("lon", "?"))
            site_id = site.get("id", site.get("site_id", "?"))
            print(f"  [{i}] ID={site_id} | {name} | ({lat}, {lon})")

        if len(sites) > 50:
            print(f"  ... 还有 {len(sites)-50} 个站点（省略显示）")

        print(f"\n  总计: {len(sites)} 个站点")
        return sites

    # ---- 数据下载 ----

    def download_pt_data(self, site_id: int) -> Optional[Path]:
        """
        下载单个站点的冻土温度数据

        Args:
            site_id: 站点 ID

        Returns:
            下载文件路径，失败返回 None
        """
        # 先获取元数据
        url = f"{GTNP_API_BASE}/pt/{site_id}"
        print(f"[{self._now()}] 📥 下载站点 {site_id} 的地温数据...")

        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()

            # 保存元数据
            meta_path = self.download_dir / f"pt_site_{site_id}_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(resp.json(), f, ensure_ascii=False, indent=2)

            # 尝试下载数据文件
            data_url = f"{GTNP_API_BASE}/data/?pt_data={site_id}&combined=false"
            resp2 = self.session.get(data_url, timeout=60)
            if resp2.status_code == 200:
                data_path = self.download_dir / f"pt_site_{site_id}_data.csv"
                with open(data_path, "wb") as f:
                    f.write(resp2.content)
                print(f"  ✅ 数据保存: {data_path}")
                self._log_download(str(data_url), data_path, len(resp2.content))
                return data_path
            else:
                print(f"  ⚠️  数据下载返回 {resp2.status_code}，元数据已保存")
                return meta_path

        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return None

    def download_alt_data(self, dataset_id: int) -> Optional[Path]:
        """
        下载活动层厚度 (ALT) 数据

        Args:
            dataset_id: 数据集 ID

        Returns:
            下载文件路径
        """
        url = f"{GTNP_API_BASE}/data/?alt_data={dataset_id}&combined=false"
        print(f"[{self._now()}] 📥 下载 ALT 数据集 {dataset_id}...")

        try:
            resp = self.session.get(url, timeout=60)
            resp.raise_for_status()

            # 可能是 zip 文件
            content_type = resp.headers.get("content-type", "")
            if "zip" in content_type or "octet-stream" in content_type:
                data_path = self.download_dir / f"alt_dataset_{dataset_id}.zip"
            else:
                data_path = self.download_dir / f"alt_dataset_{dataset_id}.csv"

            with open(data_path, "wb") as f:
                f.write(resp.content)

            size_mb = len(resp.content) / 1024 / 1024
            print(f"  ✅ 保存: {data_path} ({size_mb:.2f} MB)")
            self._log_download(str(url), data_path, len(resp.content))
            return data_path

        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return None

    def download_china_data(self):
        """批量下载中国区域所有冻土温度数据"""
        sites = self.list_sites(country="China")
        if not sites:
            print("⚠️  未找到中国区域站点，尝试直接下载...")
            return []

        results = []
        for site in sites:
            site_id = site.get("id", site.get("site_id"))
            if site_id:
                result = self.download_pt_data(int(site_id))
                if result:
                    results.append(result)
                time.sleep(1)  # 礼貌性延迟

        return results

    def download_pangaea_magt(self):
        """
        下载 PANGAEA 上的 GTN-P 全球 MAGT 数据集
        (41年 Mean Annual Ground Temperature)

        DOI: https://doi.pangaea.de/10.1594/PANGAEA.972992
        """
        print(f"[{self._now()}] 📥 下载 PANGAEA 全球 MAGT 数据集...")

        # PANGAEA 的直接下载链接格式
        # 先尝试 DOI 解析
        try:
            resp = self.session.get(PANGAEA_MAGT_URL, timeout=30, allow_redirects=True)
            # PANGAEA 页面通常有直接下载链接
            # 格式: https://doi.pangaea.de/10.1594/PANGAEA.972992?format=text
            download_url = f"https://doi.pangaea.de/10.1594/PANGAEA.972992?format=text"
            print(f"  尝试下载: {download_url}")

            resp = self.session.get(download_url, timeout=120)
            if resp.status_code == 200 and len(resp.content) > 1000:
                data_path = self.download_dir / "gtnp_magt_41years_pangaea.tsv"
                with open(data_path, "wb") as f:
                    f.write(resp.content)
                size_mb = len(resp.content) / 1024 / 1024
                print(f"  ✅ 保存: {data_path} ({size_mb:.2f} MB)")
                self._log_download(download_url, data_path, len(resp.content))
                return data_path
            else:
                print(f"  ⚠️  下载返回 {resp.status_code}，尝试备用方式...")
        except Exception as e:
            print(f"  ⚠️  下载异常: {e}")

        # 备用方式：用 wget
        print(f"  💡 备用方式：用 wget 下载")
        print(f"  运行: wget -O {self.download_dir}/gtnp_magt_41years.tsv "
              f"'https://doi.pangaea.de/10.1594/PANGAEA.972992?format=text'")
        return None

    # ---- 辅助 ----

    def _log_download(self, url: str, path: Path, size: int):
        entry = {
            "timestamp": self._now(),
            "url": url,
            "filename": path.name,
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
        }
        self.download_log.append(entry)
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.download_log, f, ensure_ascii=False, indent=2)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"📊 GTN-P 下载汇总")
        print(f"{'='*60}")
        print(f"下载目录: {self.download_dir}")
        print(f"成功下载: {len(self.download_log)} 个文件")
        total_mb = sum(e['size_mb'] for e in self.download_log)
        print(f"总大小: {total_mb:.2f} MB ({total_mb/1024:.3f} GB)")
        print(f"日志文件: {self.log_file}")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GTN-P 冻土地温数据下载脚本（无需登录，公开 API）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出中国区域站点
  python download_gtnp.py --list-china

  # 下载特定站点数据
  python download_gtnp.py --download-pt --site-id 123

  # 批量下载中国区域数据
  python download_gtnp.py --download-china

  # 下载 PANGAEA 全球 MAGT 数据
  python download_gtnp.py --download-pangaea
        """
    )

    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR,
                       help=f"下载目录（默认 {DEFAULT_DOWNLOAD_DIR}）")
    parser.add_argument("--list-china", action="store_true",
                       help="列出中国区域冻土温度站点")
    parser.add_argument("--download-pt", action="store_true",
                       help="下载指定站点的冻土温度数据")
    parser.add_argument("--site-id", type=int,
                       help="站点 ID（配合 --download-pt 使用）")
    parser.add_argument("--download-china", action="store_true",
                       help="批量下载中国区域所有冻土温度数据")
    parser.add_argument("--download-pangaea", action="store_true",
                       help="下载 PANGAEA 全球 MAGT 数据集（41年地温）")

    args = parser.parse_args()

    client = GTNPClient(download_dir=args.download_dir)

    if args.list_china:
        client.list_china_sites()

    elif args.download_pt:
        if not args.site_id:
            print("❌ 需要 --site-id 参数")
            sys.exit(1)
        client.download_pt_data(args.site_id)

    elif args.download_china:
        client.download_china_data()

    elif args.download_pangaea:
        client.download_pangaea_magt()

    else:
        parser.print_help()

    client.print_summary()


if __name__ == "__main__":
    main()
