
# 安全配置指南

本文档提供量化交易技能套件的安全配置最佳实践，确保 API 密钥和敏感信息的安全存储和使用。

## 目录

1. [环境变量安全](#环境变量安全)
2. [.gitignore 配置](#gitignore-配置)
3. [密钥管理](#密钥管理)
4. [数据库安全](#数据库安全)
5. [日志安全](#日志安全)
6. [权限控制](#权限控制)

---

## 环境变量安全

### 1. 使用 .env 文件

项目使用 `.env` 文件管理环境变量。**切勿**将 `.env` 提交到版本控制系统。

**正确做法：**
```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件填入真实配置
```

### 2. 环境变量命名规范

- 所有环境变量使用大写字母和下划线
- 敏感信息命名包含 `TOKEN`、`PASSWORD`、`SECRET`、`KEY` 等关键词
- `env_config.py` 会自动识别并隐藏这些敏感信息

敏感环境变量列表：
- `TUSHARE_TOKEN`
- `XCSC_TUSHARE_TOKEN`
- `GM_TOKEN`
- `DB_USER`
- `DB_PASSWORD`

---

## .gitignore 配置

确保项目根目录有正确的 `.gitignore` 文件，包含以下内容：

```gitignore
# 环境变量文件
.env
.env.local
.env.*.local

# 数据目录
data/
factor_data/
reports/

# 日志文件
*.log
logs/

# Python 缓存
__pycache__/
*.py[cod]
*$py.class
*.so

# IDE
.vscode/
.idea/
*.swp
*.swo

# 测试和覆盖率
.pytest_cache/
.coverage
htmlcov/
```

---

## 密钥管理

### 1. 定期轮换密钥

建议定期（如每 90 天）轮换 API 密钥，并更新 `.env` 文件。

### 2. 最小权限原则

为每个数据源申请最小必要权限的 API 密钥，避免使用管理员权限的密钥。

### 3. 密钥备份

- 将密钥备份到安全的密码管理器（如 1Password、Bitwarden）
- **切勿**通过邮件、聊天工具等明文传输密钥
- **切勿**将密钥写在代码注释或文档中

### 4. 使用环境变量而非硬编码

**错误示例：**
```python
# 切勿这样做！
token = "your_secret_token_12345"
```

**正确示例：**
```python
from env_config import get_env_config

config = get_env_config()
token = config.get_str("TUSHARE_TOKEN")
```

---

## 数据库安全

### 1. 使用 SQLite 用于开发

开发环境推荐使用 SQLite，无需配置用户和密码：

```env
DB_TYPE=sqlite
DB_NAME=a_share_data
```

### 2. 生产环境使用专用数据库用户

生产环境使用 MySQL/PostgreSQL 时：

```env
DB_TYPE=mysql
DB_HOST=your-db-host
DB_PORT=3306
DB_NAME=quant_trading
DB_USER=quant_user  # 专用用户，非 root
DB_PASSWORD=strong_password_here
```

### 3. 数据库连接加密

- 使用 SSL/TLS 加密数据库连接
- 避免在公网直接暴露数据库端口
- 限制数据库用户的访问源 IP

---

## 日志安全

### 1. 避免记录敏感信息

`env_config.py` 的 `list_all()` 方法会自动隐藏敏感信息，使用它来输出配置：

```python
from env_config import get_env_config

config = get_env_config()
print(config.list_all())  # 安全，敏感信息显示为 ***
```

### 2. 日志文件保护

- 设置日志文件权限为仅当前用户可读写
- 定期清理和归档日志文件
- 避免在日志中输出完整的 API 响应或请求

---

## 权限控制

### 1. 文件权限

设置 `.env` 文件权限为仅所有者可读：

```bash
chmod 600 .env
```

### 2. 目录权限

数据目录权限：

```bash
chmod 700 data/
chmod 700 factor_data/
chmod 700 reports/
```

### 3. 运行环境隔离

- 使用虚拟环境（venv, conda）隔离依赖
- 避免使用 root 权限运行应用程序
- 在容器中运行时遵循最小容器原则

---

## 检查配置安全性

运行环境检查脚本验证配置：

```bash
python check_env.py
```

该脚本会：
- 检查 .env 文件是否存在
- 验证必需的配置项
- 检查目录权限
- 显示配置摘要（隐去敏感信息）

---

## 应急响应

如果 API 密钥或数据库凭证泄露：

1. **立即**在对应平台上撤销泄露的密钥
2. **立即**更新数据库密码
3. 检查是否有异常数据访问
4. 轮换所有相关的密钥和密码
5. 调查泄露原因并修复安全漏洞

---

## 参考资源

- [OWASP 环境变量安全指南](https://owasp.org/)
- [12-Factor App 配置管理](https://12factor.net/config)

