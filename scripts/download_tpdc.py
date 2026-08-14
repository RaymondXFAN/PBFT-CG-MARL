"""
TPDC 国家青藏高原科学数据中心 数据下载脚本

【安全设计】
- 密码只通过环境变量 TPDC_EMAIL 和 TPDC_PWD 传入
- 脚本本身不包含任何明文密码
- 支持 Cookie 模式（从浏览器复制 Cookie，更可靠）

【TPDC 登录说明】
TPDC 网站有反爬机制，直接 POST 登录容易 405。
推荐两种方式：
1. Cookie 模式（最可靠）：浏览器登录 → F12 复制 Cookie → 脚本用 Cookie 下载
2. CKAN API 模式：TPDC 基于 CKAN，可用 API Token 下载

用法:
    # 方式1：Cookie 模式（推荐）
    # 先在浏览器里登录 https://data.tpdc.ac.cn
    # F12 → Application → Cookies → 复制 auth_tkt 的值
    python download_tpdc.py --cookie "auth_tkt=xxx" --download-urls urls.txt

    # 方式2：用环境变量自动登录（可能被反爬拦截）
    export TPDC_EMAIL="your_email"
    export TPDC_PWD="your_password"
    python download_tpdc.py --test-login

    # 方式3：直接 wget 下载（公开数据集）
    python download_tpdc.py --wget "https://data.tpdc.ac.cn/xxx.nc"

    # 列出候选数据集
    python download_tpdc.py --list
"""

import os
import sys
import json
import time
import re
import argparse
import subprocess
import requests
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse, unquote
from datetime import datetime


# ============================================================
# 配置区
# ============================================================

TPDC_BASE_URL = "https://data.tpdc.ac.cn"
DEFAULT_DOWNLOAD_DIR = "/root/autodl-tmp/tpdc_data"

# 冻土/地温相关候选数据集
PERMAFROST_DATASETS = [
    {
        "id": "high_res_permafrost_2000_2016",
        "title": "高分辨率北半球多年冻土数据集(2000-2016)",
        "metadata_url": "https://data.tpdc.ac.cn/zh-hans/data/5093d9ff-a5fc-4f10-a53f-c01e7b781368/",
        "doi": "https://doi.org/10.11888/Geocry.tpdc.271190",
        "expected_size_gb": 2.0,
    },
    {
        "id": "tibetan_plateau_permafrost_map",
        "title": "青藏高原多年冻土分布图",
        "metadata_url": "https://data.tpdc.ac.cn/zh-hans/data/60b2e5a3-5e3f-4e2d-9fb2-55a5e0e0d57b/",
        "expected_size_gb": 0.5,
    },
]


# ============================================================
# 核心客户端
# ============================================================

class TPDCClient:
    """TPDC 数据下载客户端"""

    def __init__(self, download_dir: str = DEFAULT_DOWNLOAD_DIR,
                 email: str = None, password: str = None,
                 cookie: str = None):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.password = password

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"{TPDC_BASE_URL}/zh-hans/",
        })

        self.logged_in = False
        self.download_log = []
        self.log_file = self.download_dir / "download_log.json"

        # Cookie 模式
        if cookie:
            self._set_cookie(cookie)
            self.logged_in = True

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _set_cookie(self, cookie_str: str):
        """从字符串设置 Cookie"""
        for item in cookie_str.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                self.session.cookies.set(k.strip(), v.strip())
        print(f"[{self._now()}] ✅ Cookie 已设置")

    # ---- 登录 ----

    def login(self) -> bool:
        """
        登录 TPDC（CKAN 框架）

        TPDC 的登录机制可能因反爬而变化，如果失败请用 Cookie 模式
        """
        if not self.email or not self.password:
            print(f"[{self._now()}] ❌ 缺少邮箱或密码")
            return False

        print(f"[{self._now()}] 正在登录 TPDC...")
        print(f"  邮箱: {self.email[:3]}***{self.email[-8:]}")

        try:
            # 方法1：CKAN API 登录
            # TPDC 基于 CKAN，标准 API 是 /api/3/action/
            # 先尝试获取登录页
            login_page_url = f"{TPDC_BASE_URL}/zh-hans/user/login"
            resp = self.session.get(login_page_url, timeout=30)
            if resp.status_code != 200:
                print(f"  ⚠️  登录页返回 {resp.status_code}，尝试 API 方式...")

            # 提取 CSRF token
            csrf_token = None
            # CKAN 通常在 cookie 或 hidden field 里
            if "csrf_token" in self.session.cookies:
                csrf_token = self.session.cookies["csrf_token"]
            else:
                m = re.search(
                    r'name=["\']csrf_token["\']\s*(?:value=["\']([^"\']+)["\']|[^>]*value=["\']([^"\']+)["\'])',
                    resp.text
                )
                if m:
                    csrf_token = m.group(1) or m.group(2)

            # 方法2：POST 到 CKAN 的登录端点
            # CKAN 标准登录端点
            login_endpoints = [
                f"{TPDC_BASE_URL}/zh-hans/user/login",
                f"{TPDC_BASE_URL}/user/login",
                f"{TPDC_BASE_URL}/api/3/action/user_login",
            ]

            for login_url in login_endpoints:
                try:
                    login_data = {
                        "login": self.email,
                        "password": self.password,
                        "remember": "yes",
                    }
                    if csrf_token:
                        login_data["csrf_token"] = csrf_token

                    headers = {
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Referer": login_page_url,
                        "Origin": TPDC_BASE_URL,
                    }

                    resp = self.session.post(
                        login_url,
                        data=login_data,
                        headers=headers,
                        timeout=30,
                        allow_redirects=False,
                    )

                    # 检查成功标志
                    if resp.status_code in (302, 303):
                        # 跟随重定向
                        location = resp.headers.get("Location", "")
                        if "login" not in location.lower():
                            print(f"  ✅ 登录成功（{login_url} → 302）")
                            self.logged_in = True
                            return True

                    if "auth_tkt" in self.session.cookies:
                        print(f"  ✅ 登录成功（Cookie 已设置）")
                        self.logged_in = True
                        return True

                except Exception as e:
                    print(f"  ⚠️  {login_url} 失败: {e}")
                    continue

            # 方法3：CKAN API Token（如果用户有）
            print(f"\n[{self._now()}] ❌ 自动登录失败")
            print(f"  💡 推荐使用 Cookie 模式：")
            print(f"     1. 在浏览器里打开 https://data.tpdc.ac.cn 并登录")
            print(f"     2. 按 F12 → Application → Cookies")
            print(f"     3. 复制 auth_tkt 的值")
            print(f"     4. 运行: python download_tpdc.py --cookie 'auth_tkt=复制的值' --download-urls urls.txt")
            return False

        except Exception as e:
            print(f"[{self._now()}] ❌ 登录异常: {e}")
            return False

    # ---- 下载 ----

    def download_file(self, url: str, dest_filename: str = None) -> Optional[Path]:
        """下载单个文件（带进度显示）"""
        if not dest_filename:
            parsed = urlparse(url)
            dest_filename = unquote(Path(parsed.path).name) or f"file_{int(time.time())}"

        dest_path = self.download_dir / dest_filename

        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"  ⏭️  已存在: {dest_filename} ({dest_path.stat().st_size/1024/1024:.2f} MB)")
            return dest_path

        print(f"[{self._now()}] 📥 下载: {dest_filename}")
        try:
            resp = self.session.get(url, stream=True, timeout=120, allow_redirects=True)
            resp.raise_for_status()

            total_size = int(resp.headers.get("content-length", 0))
            chunk_size = 1024 * 1024

            with open(dest_path, "wb") as f:
                downloaded = 0
                start = time.time()
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            speed = downloaded / (time.time() - start + 0.01) / 1024 / 1024
                            print(f"\r  {pct:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB) "
                                  f"{speed:.2f} MB/s", end="", flush=True)
                print()

            actual_size = dest_path.stat().st_size
            print(f"  ✅ 完成: {actual_size/1024/1024:.2f} MB")
            self._log_download(url, dest_path, actual_size)
            return dest_path

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return None

    def download_from_urls_file(self, urls_file: str) -> List[Path]:
        """从 URL 列表文件批量下载"""
        urls_file = Path(urls_file)
        if not urls_file.exists():
            print(f"❌ URL 文件不存在: {urls_file}")
            return []

        results = []
        with open(urls_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                url = parts[0].strip()
                filename = parts[1].strip() if len(parts) > 1 else None
                result = self.download_file(url, filename)
                if result:
                    results.append(result)
                time.sleep(0.5)
        return results

    def wget_download(self, url: str):
        """用 wget 下载（不受 Python 反爬限制）"""
        dest_path = self.download_dir
        print(f"[{self._now()}] 📥 用 wget 下载: {url}")
        cmd = f'wget -c -P "{dest_path}" "{url}"'
        print(f"  运行: {cmd}")
        os.system(cmd)

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
        print(f"📊 下载汇总")
        print(f"{'='*60}")
        print(f"下载目录: {self.download_dir}")
        print(f"登录状态: {'✅' if self.logged_in else '❌'}")
        print(f"成功下载: {len(self.download_log)} 个文件")
        total_mb = sum(e['size_mb'] for e in self.download_log)
        print(f"总大小: {total_mb:.2f} MB")
        print(f"日志: {self.log_file}")


# ============================================================
# 命令行
# ============================================================

def list_candidate_datasets():
    print(f"\n{'='*60}")
    print(f"📋 候选数据集（按优先级）")
    print(f"{'='*60}")
    for i, ds in enumerate(PERMAFROST_DATASETS, 1):
        print(f"\n[{i}] {ds['title']}")
        print(f"    元数据: {ds['metadata_url']}")
        if 'doi' in ds:
            print(f"    DOI: {ds['doi']}")
        print(f"    预估: {ds['expected_size_gb']} GB")
    print(f"\n💡 提示：先在浏览器登录 TPDC，在数据集页面找到下载链接")
    print(f"   然后用 --download-urls urls.txt 批量下载")


def main():
    parser = argparse.ArgumentParser(
        description="TPDC 数据下载脚本（Cookie 模式推荐）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--email", default=os.environ.get("TPDC_EMAIL"))
    parser.add_argument("--password", default=os.environ.get("TPDC_PWD"))
    parser.add_argument("--cookie", type=str,
                       help="浏览器 Cookie 字符串（推荐）")
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--list", action="store_true",
                       help="列出候选数据集")
    parser.add_argument("--test-login", action="store_true",
                       help="测试登录")
    parser.add_argument("--url", type=str,
                       help="下载单个 URL")
    parser.add_argument("--wget", type=str,
                       help="用 wget 下载（不受反爬限制）")
    parser.add_argument("--download-urls", type=str,
                       help="从 URL 列表文件批量下载")

    args = parser.parse_args()

    if args.list:
        list_candidate_datasets()
        return

    # 创建客户端
    client = TPDCClient(
        download_dir=args.download_dir,
        email=args.email,
        password=args.password,
        cookie=args.cookie,
    )

    if args.test_login:
        if client.login():
            print("✅ 登录成功")
        else:
            print("❌ 登录失败，请用 Cookie 模式")
        return

    if args.url:
        client.download_file(args.url)

    if args.wget:
        client.wget_download(args.wget)

    if args.download_urls:
        client.download_from_urls_file(args.download_urls)

    client.print_summary()


if __name__ == "__main__":
    main()
