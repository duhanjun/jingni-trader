import datetime
import time
import requests
import pandas
import numpy
import pymysql
from pymysql import OperationalError

# 企业微信消息推送
class WechatNotifier:
    # 全局缓存，以SECRET为键隔离存储不同应用的Token结构: {SECRET: {"token": str, "expires_at": float}}
    _TOKEN_CACHE = {}
    
    def __init__(self, corp_id, secret, agent_id):
        self.CORP_ID = corp_id
        self.SECRET = secret
        self.AGENT_ID = agent_id
        
        # 创建Session
        self.session = requests.Session()
        self._refresh_token_if_needed()
    
    def _get_cache_key(self):
        # 唯一缓存键，使用SECRET标识独立应用
        return self.SECRET
    
    def _is_token_valid(self, cache_key):
        # 检查当前应用的Token是否有效
        cache = self._TOKEN_CACHE.get(cache_key, {})
        return time.time() < cache.get("expires_at", 0) and cache.get("token")
    
    def _refresh_token_if_needed(self):
        # 若缓存无效则刷新Token
        cache_key = self._get_cache_key()
        if not self._is_token_valid(cache_key):
            self._refresh_token(cache_key)
    
    def _retry_request(self, request_func, max_retries=3):
        # 访问请求重试
        for i in range(max_retries):
            try:
                response = request_func()
                if response is not None:
                    return response
            except Exception as e:
                print("企业微信请求错误(尝试 %s/%s): %s" % (i+1, max_retries, str(e)))
                time.sleep(0.5 * (i+1))
            
            # 如果已达到最大重试次数
            if i == max_retries - 1:
                print("企业微信请求失败，已达到最大重试次数")
                return None
        
        return None
    
    def _refresh_token(self, cache_key):
        # 请求新 Token 并更新缓存
        url = "https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=%s&corpsecret=%s" % (self.CORP_ID, self.SECRET)
        
        def token_request():
            try:
                resp = self.session.get(url, verify=False, timeout=5).json()
            except Exception as e:
                print("企业微信Token刷新请求失败: %s" % str(e))
                return None
            
            if resp.get("errcode") == 0:
                expires_in = resp.get("expires_in", 7200)
                self._TOKEN_CACHE[cache_key] = {
                    "token": resp["access_token"],
                    "expires_at": time.time() + expires_in - 300
                }
                return True
            else:
                errmsg = resp.get("errmsg", "未知错误")
                print("企业微信Token获取失败: %s" % errmsg)
                return False
        
        # 使用访问请求重试函数
        return self._retry_request(token_request)
    
    def send_qywx_text(self, content, toparty, touser):
        # 发送消息（自动使用隔离的Token）
        cache_key = self._get_cache_key()
        token_data = self._TOKEN_CACHE.get(cache_key, {})
        token = token_data.get("token")
        
        if not token:
            print("企业微信发送失败: 无效的Token")
            return None
        
        url = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s" % token
        
        payload = {
            "toparty": toparty,
            "touser": touser,
            "msgtype": "text",
            "agentid": self.AGENT_ID,
            "text": {"content": content},
            "safe": 0
        }
        
        def message_request():
            try:
                resp = self.session.post(url, json=payload, verify=False, timeout=5).json()
            except Exception as e:
                print("企业微信消息发送网络错误: %s" % str(e))
                return None
            
            return resp
        
        # 使用访问请求重试函数
        resp = self._retry_request(message_request)
        if resp is None:
            return None
        
        # 检查API响应
        errcode = resp.get("errcode", -1)
        if errcode == 0:
            return resp
        
        errmsg = resp.get("errmsg", "未知错误")
        
        # Token失效时自动刷新重试
        if errcode in [40014, 42001]:
            print("企业微信Token已失效(%s), 尝试刷新并重发" % errcode)
            self._refresh_token(cache_key)
            return self.send_qywx_text(content, toparty, touser)
        else:
            print("企业微信消息发送失败(代码%s): %s" % (errcode, errmsg))
            return resp

# 盘前执行，管理策略回购记录避免重复委托
class StockPortfolio:
    def __init__(self):
        self.traded_stocks = set()

    def add_traded_stock(self, order_id):
        self.traded_stocks.add(order_id)

    def has_traded_stock(self, order_id):
        return order_id in self.traded_stocks

# pymysql建立MySQL连接函数
def pymysql_connect():

    while True:

        # 连接MySQL数据库
        try:
            connection = pymysql.connect(
                host='citongshuo.mysql.rds.aliyuncs.com',
                user='jingni_analysis_system',
                password='wochong910119#',
                db='jingni',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor)
            # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '连接MySQL数据库成功')
            return connection

        # 捕获连接MySQL数据库超时错误
        except OperationalError as e:
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '连接MySQL数据库超时，正在尝试重新连接...')
            time.sleep(5)
            continue

# pymysql提交MySQL事务函数
def pymysql_sql(sql):

    connection=pymysql_connect()
    if connection is None:  
        return None 

    try:
        with connection.cursor() as cursor:
            # 执行 SQL 插入语句
            cursor.execute(sql)
            # 提交事务
            connection.commit()

    except pymysql.Error as e: 
        print("执行SQL时出错: %s" % (e))  
        # 出现错误时进行回滚以保持数据库一致性  
        connection.rollback()  
        return None

    finally:
        # 关闭游标  
        cursor.close()
        # 关闭数据库连接
        connection.close()

    return cursor.fetchall() if cursor else None 

# pymysql提交MySQL事务函数
def pymysql_sql_values(sql, values):

    connection=pymysql_connect()
    if connection is None:  
        return None 

    try:
        with connection.cursor() as cursor:            
            # 执行 SQL 插入语句  
            cursor.execute(sql, values)
            # 提交事务
            connection.commit()

    except pymysql.Error as e: 
        print("执行SQL时出错: %s" % (e))  
        # 出现错误时进行回滚以保持数据库一致性  
        connection.rollback()  
        return None

    finally:
        # 关闭游标
        cursor.close()
        # 关闭数据库连接
        connection.close()

    return cursor.fetchall() if cursor else None

# 证券代码转换函数
def securities_code_conversion(securities_code):

    # 获取证券代码.之前的字符串,即六位数字字符
    securities_code_char_number = securities_code[0:securities_code.rfind('.')]
    # 获取证券代码.开始的字符串，即点和市场字符
    securities_code_char_market =securities_code[securities_code.rfind('.'):]
    # 如果市场字符等于'.SH'则将市场字符替换成'.SS'
    if securities_code_char_market == '.SH':
        securities_code_char_market = '.SS'
    # 如果市场字符等于'.SZ'则将市场字符替换成'.SZ'
    elif securities_code_char_market == '.SZ':
        securities_code_char_market = '.SZ'
    # 如果市场字符等于'.SS'则将市场字符替换成'.SH'
    elif securities_code_char_market == '.SS':
        securities_code_char_market = '.SH'
    # 如果市场字符等于'.SZ'则将市场字符替换成'.SZ'
    elif securities_code_char_market == '.SZ':
        securities_code_char_market = '.SZ'
    # 如果市场字符等于'.XSHG'则将市场字符替换成'.SH'
    elif securities_code_char_market == '.XSHG':
        securities_code_char_market = '.SS'
    # 如果市场字符等于'.XSHE'则将市场字符替换成'.SZ'
    elif securities_code_char_market == '.XSHE':
        securities_code_char_market = '.SZ'
    # 将六位数字字符和点和市场字符重新拼接成新的证券代码
    securities_code = securities_code_char_number + securities_code_char_market
    return securities_code

# 获取PTrade交易日历
def ptrade_trade_cal(trade_day_count):

    # 调用get_trading_day接口数据，trade_day_count为倒数天数，0代表今天，-1代表前一天，以此类推
    ptrade_trade_cal_data = get_trading_day(trade_day_count).strftime("%Y%m%d")

    return ptrade_trade_cal_data

# 获取PTrade指定交易日历
def ptrade_trade_days(trade_start_date, trade_end_date, trade_day_count):

    # 调用get_trade_days接口数据，trade_day_count必须大于0，表示获取end_date往前的count个交易日，包含end_date当天
    trade_days_array = get_trade_days(start_date=trade_start_date, end_date=trade_end_date, count=trade_day_count)
    
    # 获取最早的交易日（数组中的第一个元素）
    if len(trade_days_array) > 0:
        earliest_trade_day = trade_days_array[0]
        
        # 检查返回的数据类型，如果是字符串，直接返回
        if isinstance(earliest_trade_day, str):
            # 如果已经是YYYY-MM-DD格式，转换为YYYYMMDD格式
            if '-' in earliest_trade_day:
                ptrade_trade_days_data = earliest_trade_day.replace('-', '')
            else:
                ptrade_trade_days_data = earliest_trade_day
        else:
            # 如果是日期对象，使用strftime格式化
            ptrade_trade_days_data = earliest_trade_day.strftime("%Y%m%d")
    else:
        # 如果没有交易日数据，返回空字符串
        ptrade_trade_days_data = ""

    return ptrade_trade_days_data

# 策略启动消息函数
def strategy_start_send_qywx(context):

    log.info('开始企微消息推送') 
    # send_qywx('ww137c94e7a2caf6fb', 'AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I', '1000003', info='启动%s' % g.__strategy_name, touser= 'duhanjun')
    WechatNotifier("ww137c94e7a2caf6fb", "AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I", "1000003").send_qywx_text('启动%s' % g.__strategy_name, '', 'duhanjun')
    log.info('完成企微消息推送')

# 新股新债申购函数
def ipo_security(context):

    log.info('开始新股新债申购')

    # 上证新股申购
    # log.info("申购上证新股")
    # stock_ipo_sh = ipo_stocks_order(market_type=0)
    # time.sleep(5)

    # 上证科创板新股申购
    # log.info("申购上证科创板新股")
    # stock_ipo_sh_kc = ipo_stocks_order(market_type=1)
    # time.sleep(5)

    # 深证新股申购
    # log.info("申购深证新股")
    # stock_ipo_sz = ipo_stocks_order(market_type=2)
    # time.sleep(5)

    # 深证创业板新股申购
    # log.info("申购深证创业板新股")
    # stock_ipo_sh_cy = ipo_stocks_order(market_type=3)
    # time.sleep(5)

    # 沪深可转债新债申购
    debt_ipo = ipo_stocks_order(market_type=4)
    time.sleep(5)

    # 通过keys函数获取debt_ipo字典内的所有key值(即可转债代码)并转换成list列表形式，然后通过for-in语句获取list列表中的全部可转债代码(即debt_ipo_code)
    for debt_ipo_code in list(debt_ipo.keys()):
        # send_qywx('ww137c94e7a2caf6fb', 'AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I', '1000003', info='%s：\n申购沪深可转债：%s' % (g.__strategy_name, debt_ipo_code), touser= 'duhanjun')
        WechatNotifier("ww137c94e7a2caf6fb", "AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I", "1000003").send_qywx_text('%s：\n申购沪深可转债：%s' % (g.__strategy_name, debt_ipo_code), '', 'duhanjun')

    log.info('完成新股新债申购')

# 国债逆回购函数
def treasury_reverse_repurchase(context):

    log.info('开始国债逆回购')

    # 通过for-in语句获取账户当日委托的全部订单
    for _order in get_all_orders():

        # 如果当前已委托订单(entrust_bs取值为1)中，有处于已报状态(status取值为2)和部成状态(status取值为7)的订单，则取消这些订单
        if (_order['entrust_bs']==1) & (_order['status'] in ['2', '7']):
            cancel_order_ex(_order)

    # 获取可用资金
    cash = context.portfolio.cash

    # 获取证券代码
    stock_code = '204001.SS'

    # 获取证券名称
    stock_name_dict = get_stock_name(stock_code)
    stock_name = stock_name_dict[stock_code]

    # 计算证券可卖数量
    stock_sell_amount = (int(cash/1000))*10

    # 计算证券可卖金额
    stock_sell_money = 100*stock_sell_amount

    # 如果证券可卖数量大于0
    if stock_sell_amount > 0:
        # 根据证券可卖数量卖出证券
        order(stock_code, -stock_sell_amount)

    # 如果证券可卖数量等于0
    else:
        pass

    log.info('完成国债逆回购')

# 策略持仓消息函数
def strategy_position_send_qywx(context):

    log.info('开始策略持仓消息推送')

    # 如果业务场景是交易
    if is_trade():

        # 获取当前持有的标的(包含不可卖出的标的)，dict类型，key是证券代码，value是Position对象
        positions = context.portfolio.positions

        for security_code in positions:

            # 通过key证券代码获取当前持有标的对应的Position对象
            position = positions[security_code]

            security_name = get_stock_name(security_code)[security_code]
            # print('证券名称：%s'% (security_name))

            security_code = securities_code_conversion(security_code)
            # print('证券代码：%s' % (security_code))

            security_cost_basis = float(position.cost_basis)
            # print('成本价格：%s' % (security_cost_basis))

            security_last_sale_price = float(position.last_sale_price)
            # print('最新价格：%s' % (security_last_sale_price))

            security_returns = float(round(((security_last_sale_price - security_cost_basis)/security_cost_basis) * 100, 2))
            # print('浮动盈亏：%s' % (security_returns))

            security_amount = position.amount
            portfolio_value = context.portfolio.portfolio_value
            security_portfolio_ratio = float(round(((security_last_sale_price*security_amount)/portfolio_value * 100), 2))
            # print('证券仓位：%s' % (security_portfolio_ratio))

            # send_qywx('ww137c94e7a2caf6fb', 'n0UdA9YmkTbe-L-X_cj51axFYHdgXXThMPVJ_kNGRmw', '1000004', info='%s：\n证券名称：%s\n证券代码：%s\n成本价格：%s元\n最新价格：%s元\n浮动盈亏：%s%%\n证券仓位：%s%%' % (g.__strategy_name, security_name, security_code, security_cost_basis, security_last_sale_price, security_returns, security_portfolio_ratio), toparty= '7|11')
            WechatNotifier("ww137c94e7a2caf6fb", "n0UdA9YmkTbe-L-X_cj51axFYHdgXXThMPVJ_kNGRmw", "1000004").send_qywx_text('%s：\n证券名称：%s\n证券代码：%s\n成本价格：%s元\n最新价格：%s元\n浮动盈亏：%s%%\n证券仓位：%s%%' % (g.__strategy_name, security_name, security_code, security_cost_basis, security_last_sale_price, security_returns, security_portfolio_ratio), '7|11', '')

            time.sleep(5)
    # 如果业务场景是回测
    else:
        pass

    log.info('完成策略持仓消息推送')

# 策略组合消息函数
def strategy_portfolio_send_qywx(context):

    log.info('开始策略日报推送')

    # 如果业务场景是交易
    if is_trade():

        # 获取当日账户资产总和并保留两位小数
        assets_value = g.assets_value
        # log.info('资产总和：%s元' % assets_value)

        # 获取当日账户市值总和并保留两位小数
        market_value = g.market_value
        # log.info('市值总和：%s元' % market_value) 

        # 获取当前账户的持仓比例并保留两位小数
        assets_market_ratio = g.assets_market_ratio
        # log.info("仓位比例：%s%%" % (assets_market_ratio))

        # 获取当日账户可用现金并保留两位小数
        cash = g.cash
        # log.info('可用现金：%s元' % cash)

        # send_qywx('ww137c94e7a2caf6fb', 'AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I', '1000003', info='%s：\n资产总和：%s元\n市值总和：%s元\n仓位比例：%s%%\n可用现金：%s元' % (g.__strategy_name, assets_value,market_value,assets_market_ratio,cash), touser= 'duhanjun')
        WechatNotifier("ww137c94e7a2caf6fb", "AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I", "1000003").send_qywx_text('%s：\n资产总和：%s元\n市值总和：%s元\n仓位比例：%s%%\n可用现金：%s元' % (g.__strategy_name, assets_value,market_value,assets_market_ratio,cash), '', 'duhanjun')

    # 如果业务场景是回测
    else:
        pass

    log.info('完成策略日报推送')

# 策略订单委托函数
def stock_order_value(context, stock_code, stock_value):

    if stock_value > 0:

        # 获取账户可用资金
        cash = g.cash
        log.info("%s 获取账户可用资金：%s" % (stock_code,cash))

        # 如果业务场景是交易
        if is_trade():
            # 获取证券今日的最新价-实盘环境
            stock_snapshot = get_snapshot(stock_code)
            # 判断证券今日的最新价是否为空
            if 'last_px' not in stock_snapshot[stock_code]:
                # 设定证券今日的最新价缺省值
                stock_current_price = 0
            else:
                # 获取证券今日的最新价
                stock_current_price = stock_snapshot[stock_code]['last_px']
            print("%s 获取证券最新价格：%s" % (stock_code, stock_current_price))
        # 如果业务场景是回测
        else:
            # 获取证券今日的收盘价-回测环境
            stock_history_today = get_history(1, '1d', 'close', stock_code, fq='pre', include=True)
            # 判断证券今日的收盘价是否为空
            if stock_history_today.empty:
                # 设定证券今日的收盘价缺省值
                stock_current_price = 0
            else:
                # 获取证券今日的收盘价
                stock_current_price = stock_history_today['close'][-1]
            print("%s 获取证券收盘价格：%s" % (stock_code, stock_current_price))

        if stock_value <= cash:
            # 用委托资金买入证券
            order_value(stock_code, stock_value, limit_price=stock_current_price*1.0030)
            log.info("委托资金买入证券")

        elif stock_value > cash:
            # 用可用资金买入证券
            order_value(stock_code, cash, limit_price=stock_current_price*1.0030)
            log.info("可用资金买入证券")

    elif stock_value < 0:

        # 获取证券可卖数量
        stock_enable_amount = -(get_position(stock_code).enable_amount)
        log.info("%s 获取证券可卖数量：%s" % (stock_code, -stock_enable_amount))

        # 如果业务场景是交易
        if is_trade():
            # 获取证券今日的最新价-实盘环境
            stock_snapshot = get_snapshot(stock_code)
            # 判断证券今日的最新价是否为空
            if 'last_px' not in stock_snapshot[stock_code]:
                # 设定证券今日的最新价缺省值
                stock_current_price = 0
            else:
                # 获取证券今日的最新价
                stock_current_price = stock_snapshot[stock_code]['last_px']
            print("%s 获取证券最新价格：%s" % (stock_code, stock_current_price))
        # 如果业务场景是回测
        else:
            # 获取证券今日的收盘价-回测环境
            stock_history_today = get_history(1, '1d', 'close', stock_code, fq='pre', include=True)
            # 判断证券今日的收盘价是否为空
            if stock_history_today.empty:
                # 设定证券今日的收盘价缺省值
                stock_current_price = 0
            else:
                # 获取证券今日的收盘价
                stock_current_price = stock_history_today['close'][-1]
            print("%s 获取证券收盘价格：%s" % (stock_code, stock_current_price))

        # 计算证券委托数量
        stock_amount = (int(stock_value/(stock_current_price*100))*100)
        log.info("%s 证券卖出委托数量：%s" % (stock_code, -stock_amount))

        if stock_amount >= stock_enable_amount and stock_enable_amount < 0:
            # 用委托数量卖出证券
            order(stock_code, stock_amount, limit_price=stock_current_price*0.9970)
            log.info("委托数量卖出证券")

        elif stock_amount < stock_enable_amount and stock_enable_amount < 0:
            # 用可卖数量卖出证券
            order(stock_code, stock_enable_amount, limit_price=stock_current_price*0.9970)
            log.info("可卖数量卖出证券")

# 成交主推回调函数
def on_trade_response(context, trade_list):

    log.info('开始执行成交主推回调函数')

    # 如果业务场景是交易
    if is_trade():

        trade_list = trade_list
        log.info('成交日志：%s'% (trade_list))

        trading_datetime = trade_list[0]["business_time"]
        trading_datetime = datetime.datetime.strptime(trading_datetime, '%Y-%m-%d %H:%M:%S')
        log.info('交易日期：%s'% (trading_datetime))

        order_id = trade_list[0]["order_id"]+'.'+trade_list[0]["business_id"]
        log.info('订单编号：%s'% (order_id))

        security_code = trade_list[0]["stock_code"]

        security_name = get_stock_name(security_code)[security_code]
        log.info('证券名称：%s'% (security_name))

        securities_code = security_code
        # 获取证券代码.之前的字符串,即六位数字字符
        securities_code_char_number = securities_code[0:securities_code.rfind('.')]
        # 获取证券代码.开始的字符串，即点和市场字符
        securities_code_char_market =securities_code[securities_code.rfind('.'):]
        # 如果市场字符等于'.XSHG'则将市场字符替换成'.SH'
        if securities_code_char_market == '.SS':
            securities_code_char_market = '.SH'
        # 如果市场字符等于'.SZ'则将市场字符替换成'.SZ'
        elif securities_code_char_market == '.SZ':
            securities_code_char_market = '.SZ'
        # 将六位数字字符和点和市场字符重新拼接成新的证券代码
        security_code = securities_code_char_number + securities_code_char_market
        log.info('证券代码：%s'% (security_code))

        trading_entrust_direction = trade_list[0]["entrust_bs"]
        log.info('交易方向：%s'% (trading_entrust_direction))

        trading_amount = trade_list[0]["business_amount"]
        log.info('交易数量：%s'% (trading_amount))

        trading_price = float(trade_list[0]["business_price"])
        log.info('交易价格：%s'% (trading_price))

        trading_total_money = trade_list[0]["business_balance"]
        log.info('交易金额：%s'% (trading_total_money))

        if trading_total_money < 0:
            trading_total_money = -trading_total_money

        trading_trading_fee = float(trading_total_money*0.0001)
        log.info('交易费用：%s'% (trading_trading_fee))

        security_assets_ratio = float(round(trading_total_money/context.portfolio.portfolio_value, 4))
        log.info('证券仓位：%s'% (security_assets_ratio))

        # 定义MySQL数据库写入语句  
        strategy_trading_sql = "INSERT INTO strategy_trading (交易日期, 订单编号, 策略名称, 证券代码, 证券名称, 交易方向, 交易数量, 交易价格, 交易金额, 交易费用, 证券仓位, 指数代码, 指数名称) SELECT %s AS 交易日期, %s AS 订单编号, %s AS 策略名称, %s AS 证券代码, %s AS 证券名称, %s AS 交易方向, %s AS 交易数量, %s AS 交易价格, %s AS 交易金额, %s AS 交易费用, %s AS 证券仓位, 指数代码, 指数名称 FROM jingni_index_data WHERE 基金代码 = %s LIMIT 0,1 ON DUPLICATE KEY UPDATE 交易日期 = VALUES(交易日期), 订单编号 = VALUES(订单编号), 策略名称 = VALUES(策略名称), 证券代码 = VALUES(证券代码), 证券名称 = VALUES(证券名称), 交易方向 = VALUES(交易方向), 交易数量 = VALUES(交易数量), 交易价格 = VALUES(交易价格), 交易金额 = VALUES(交易金额), 交易费用 = VALUES(交易费用), 证券仓位 = VALUES(证券仓位)"
        # 定义MySQL数据库写入参数
        strategy_trading_values = (trading_datetime, order_id, g.__strategy_name, security_code, security_name, trading_entrust_direction, trading_amount, trading_price, trading_total_money, trading_trading_fee, security_assets_ratio, security_code)    
        # 执行MySQL数据库写入函数
        pymysql_sql_values(strategy_trading_sql, strategy_trading_values)

        # send_qywx('ww137c94e7a2caf6fb', 'n0UdA9YmkTbe-L-X_cj51axFYHdgXXThMPVJ_kNGRmw', '1000004', info='%s：\n证券名称：%s\n证券代码：%s\n交易方向：%s(1买入|2卖出)\n证券仓位：%s' % (g.__strategy_name, security_name, security_code, trading_entrust_direction, security_assets_ratio), toparty= '7|11')
        WechatNotifier("ww137c94e7a2caf6fb", "n0UdA9YmkTbe-L-X_cj51axFYHdgXXThMPVJ_kNGRmw", "1000004").send_qywx_text('%s：\n证券名称：%s\n证券代码：%s\n交易方向：%s(1买入|2卖出)\n证券仓位：%s' % (g.__strategy_name, security_name, security_code, trading_entrust_direction, security_assets_ratio), '7|11', '')

        time.sleep(5)

    # 如果业务场景是回测
    else:
        pass

    log.info('完成执行成交主推回调函数')

# 策略账户信息函数
def portfolio(context):

    # 获取当前账户的资产总和
    g.assets_value = context.portfolio.portfolio_value
    # log.info("资产总和：%s元" % (g.assets_value))

    # 获取当前账户的市值总和
    g.market_value = context.portfolio.positions_value
    # log.info("市值总和：%s元" % (g.market_value))

    # 获取当前账户的持仓比例并保留两位小数
    g.assets_market_ratio = round(g.market_value/g.assets_value, 4) * 100
    # log.info("仓位比例：%s%%" % (g.assets_market_ratio))

    # 获取当前账户的可用资金
    g.cash = context.portfolio.cash
    # log.info("可用资金：%s元" % (g.cash))

# 策略持仓更新函数
def position(context):

    # 如果业务场景是交易
    if is_trade():

        # 获取当前持有的标的(包含不可卖出的标的)，dict类型，key是证券代码，value是Position对象
        positions = context.portfolio.positions

        # 创建证券代码空列表
        security_code_list = []

        for security_code in positions:

            # 通过key证券代码获取当前持有标的对应的Position对象
            position = positions[security_code]

            security_name = get_stock_name(security_code)[security_code]
            # print('证券名称：%s'% (security_name))

            security_code = securities_code_conversion(security_code)
            # print('证券代码：%s' % (security_code))

            security_cost_basis = float(position.cost_basis)
            # print('成本价格：%s' % (security_cost_basis))

            security_last_sale_price = float(position.last_sale_price)
            # print('最新价格：%s' % (security_last_sale_price))

            security_returns = float(round((security_last_sale_price - security_cost_basis)/security_cost_basis, 5))
            # print('浮动盈亏：%s' % (security_returns))

            security_amount = position.amount
            portfolio_value = context.portfolio.portfolio_value
            security_portfolio_ratio = float(round((security_last_sale_price*security_amount)/portfolio_value, 3))
            # print('证券仓位：%s' % (security_portfolio_ratio))

            # 定义MySQL数据库写入语句
            position_sql = "INSERT INTO position (策略名称, 证券名称, 证券代码, 证券仓位, 成本价格, 最新价格, 浮动盈亏) VALUES (%s, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE 证券仓位=VALUES(证券仓位), 成本价格=VALUES(成本价格), 最新价格=VALUES(最新价格), 浮动盈亏=VALUES(浮动盈亏)"
            # 定义MySQL数据库写入参数
            position_values = (g.__strategy_name, security_name, security_code, security_portfolio_ratio, security_cost_basis, security_last_sale_price, security_returns)    
            # 执行MySQL数据库写入函数
            pymysql_sql_values(position_sql, position_values)

            # 将证券代码加入列表
            security_code_list.append(security_code)

        if security_code_list:
            # 定义MySQL数据库写入语句
            security_code_list_sql = "DELETE FROM position WHERE 证券代码 NOT IN (" + ", ".join(map(lambda code: "'" + code + "'", security_code_list)) + ")" + "AND 策略名称 =%s"

        else:
            # 定义MySQL数据库写入语句
            security_code_list_sql = "DELETE FROM position WHERE 策略名称 =%s"

        # 定义MySQL数据库写入参数
        security_code_list_values = (g.__strategy_name)  
        # 执行MySQL数据库写入函数
        pymysql_sql_values(security_code_list_sql, security_code_list_values)

    # 如果业务场景是回测
    else:
        pass

# 策略资金权重函数
def get_order_weight(jingni_score_today,assets_market_ratio):

    # 定义证券交易权重初始值为0，避免if语句匹配无结果函数返回order_weight时报错
    order_weight = 0

    # 如果惊泥指数今日得分大于等于0，小于500，则证券买入时最高100%仓位
    if jingni_score_today >= 0 and jingni_score_today < 500:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2
        elif assets_market_ratio <= 65.00:
            order_weight = 0.2
        elif assets_market_ratio <= 85.00:
            order_weight = 0.2*0.995

    # 如果惊泥指数今日得分大于等于500，小于600，则证券买入时最高90%仓位
    elif jingni_score_today >= 500 and jingni_score_today < 600:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2
        elif assets_market_ratio <= 65.00:
            order_weight = 0.2
        elif assets_market_ratio <= 85.00:
            order_weight = 0.1

    # 如果惊泥指数今日得分大于等于600，小于700，则证券买入时最高80%仓位
    elif jingni_score_today >= 600 and jingni_score_today < 700:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2
        elif assets_market_ratio <= 65.00:
            order_weight = 0.2

    # 如果惊泥指数今日得分大于等于700，小于800，则证券买入时最高70%仓位
    elif jingni_score_today >= 700 and jingni_score_today < 800:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2
        elif assets_market_ratio <= 65.00:
            order_weight = 0.1

    # 如果惊泥指数今日得分大于等于800，小于900，则证券买入时最高60%仓位
    elif jingni_score_today >= 800 and jingni_score_today < 900:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2

    # 如果惊泥指数今日得分大于等于900，小于1000，则证券买入时最高50%仓位
    elif jingni_score_today >= 900 and jingni_score_today < 1000:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.1

    # 如果惊泥指数今日得分大于等于1000，小于3000，则证券买入时最高100%仓位
    elif jingni_score_today >= 1000 and jingni_score_today < 3000:
        if assets_market_ratio == 0:
            order_weight = 0.2
        elif assets_market_ratio <= 15.00:
            order_weight = 0.2
        elif assets_market_ratio <= 45.00:
            order_weight = 0.2
        elif assets_market_ratio <= 65.00:
            order_weight = 0.2
        elif assets_market_ratio <= 85.00:
            order_weight = 0.2*0.995

    return order_weight

# 转换证券代码后缀字符函数
def security_code_conversion(security_code):

    # 获取证券代码.之前的字符串,即前缀字符
    security_code_char_number = security_code[0:security_code.rfind('.')]
    # 获取证券代码.开始的字符串，即点和后缀字符
    security_code_char_market =security_code[security_code.rfind('.'):]
    # 如果后缀字符等于'.SWI'则将后缀字符替换成'.SWI'
    if security_code_char_market == '.SWI':
        security_code_char_market = '.SL'
        # 如果后缀字符等于'.SL'则将后缀字符替换成'.SWI'
    elif security_code_char_market == '.SL':
        security_code_char_market = '.SWI'
    # 如果后缀字符等于'.SH'则将后缀字符替换成'.SS'
    elif security_code_char_market == '.SH':
        security_code_char_market = '.SS'
    # 如果后缀字符等于'.SS'则将后缀字符替换成'.SH'
    elif security_code_char_market == '.SS':
        security_code_char_market = '.SH'
    # 将前缀字符和点和后缀字符重新拼接成新的证券代码
    security_code = security_code_char_number + security_code_char_market

    return security_code

# 转换证券代码前缀小写函数
def security_code_letter_lower(security_code):

    # 获取证券代码.之前的字符串,即前缀字符，并将其中的英文字母格式化成小写字母
    security_code_char_number = security_code[0:security_code.rfind('.')].lower()
    # 获取证券代码.开始的字符串，即点和后缀字符
    security_code_char_market =security_code[security_code.rfind('.'):]
    # 将前缀字符和点和后缀字符重新拼接成新的证券代码
    security_code = security_code_char_number + security_code_char_market

    return security_code

# 转换证券代码前缀大写函数
def security_code_letter_upper(security_code):

    # 获取证券代码.之前的字符串,即前缀字符，并将其中的英文字母格式化成大写字母
    security_code_char_number = security_code[0:security_code.rfind('.')].upper()
    # 获取证券代码.开始的字符串，即点和后缀字符
    security_code_char_market =security_code[security_code.rfind('.'):]
    # 将前缀字符和点和后缀字符重新拼接成新的证券代码
    security_code = security_code_char_number + security_code_char_market

    return security_code

# 获取指数证券代码列表
def get_index_list(sql):

    # 查询MySQL数据库数据
    index_list_data = pymysql_sql(sql)

    # 创建指数空列表
    index_list = []

    for index_code in index_list_data:
        # 取出index_list_data中的每个证券代码字符串
        index_code = index_code['指数代码']
        # 将处理后的证券代码字符串逐个加入指数空列表
        index_list.append(index_code)

    return index_list

# 获取指数成分股证券代码列表
def get_stock_list(sql):

    # 查询MySQL数据库数据
    stock_list_data = pymysql_sql(sql)

    # 检查获取的数据是否为空
    if stock_list_data is not None:

        # 创建股票空列表
        stock_list = []
    
        for index_code in stock_list_data:
            # 取出stock_list_data中的每个证券代码字符串
            index_code = index_code['p03473_f002']
            # 将处理后的证券代码字符串逐个加入股票空列表
            stock_list.append(index_code)
    
        return stock_list

# 计算简单移动平均线MA
def get_ma_day(stock_days,stock_close_array,stock_current_price):

    # 计算证券最近N-1个交易日的收盘价之和
    stock_close_sum = stock_close_array[-(stock_days-1):].sum()
    # 计算证券MA均线价格
    ma = (stock_current_price + stock_close_sum)/stock_days
    # 返回证券MA均线价格
    return ma

# 计算趋势得分因子数据
def jingni_qsdf_factor(index_list, trade_date_count):

    # 获取最新交易时间
    trade_now_time = datetime.datetime.now().time()

    # 如果最新交易时间大于策略开始执行时间小于策略结束执行时间
    if trade_now_time <= g.trade_end_time:

        log.info("趋势得分因子执行时间：%s" % (trade_now_time))

        # # 每次执行创建一个空的DataFrame
        # jingni_qsdf_factor_df = pandas.DataFrame(columns=['日期', '指数代码', '趋势得分'])

        # 获取策略计算的指数列表
        jingni_index_list = index_list

        for jingni_index_code in jingni_index_list:

            # 获取指数成分列表 
            stock_code_list = g.jingni_index_security_code_list[g.jingni_index_security_code_list['指数代码'] == jingni_index_code]['股票代码'].tolist()
            # print(jingni_index_code,'指数成分列表:')

            # 计算指数成分股数量
            index_member_stock_sum = len(stock_code_list)
            # print(jingni_index_code, '指数成分股票数量:', index_member_stock_sum)

            # 定义指数有效股票数量
            index_member_stock_valid = 0
            # print(jingni_index_code, '指数有效股票数量:', index_member_stock_valid)

            # 定义指数无效股票数量
            index_member_stock_invalid = 0
            # print(jingni_index_code, '指数无效股票数量:', index_member_stock_invalid)

            for stock_code in stock_code_list:

                # 通过security_code_conversion函数转换证券代码
                stock_code = security_code_conversion(stock_code)

                # 获取证券最近十九日的收盘价数据
                # stock_history = get_history(20, '1d', 'close', stock_code, fq='pre', include=False)
                stock_history = get_price(stock_code, start_date=ptrade_trade_days(None, ptrade_trade_cal(trade_date_count), 20), end_date=ptrade_trade_cal(trade_date_count), frequency='1d', fields='close', fq='pre', count=None)
                # print(stock_code, '最近十九日的收盘价数据:', stock_history)

                # 判断证券最近十九日的收盘价数据是否非空
                if stock_history.empty:
                    stock_current_price = 0
                    stock_ma20 = 0
                else:
                    # 定义证券最近十九日的收盘价数组
                    stock_close_array = stock_history['close'].values
                    # print(stock_close_array)

                    # 如果业务场景是交易
                    if is_trade():
                        # 获取证券今日的最新价-实盘
                        stock_snapshot = get_snapshot(stock_code)
                        # 判断证券今日的最新价是否为空
                        if 'last_px' not in stock_snapshot[stock_code]:
                            # 设定证券今日的最新价缺省值
                            stock_current_price = 0
                        else:
                            # 获取证券今日的最新价
                            stock_current_price = stock_snapshot[stock_code]['last_px']
                            # print(stock_code, '今日的收盘价:', stock_current_price)
                    # 如果业务场景是回测
                    else:
                        # 获取证券今日的收盘价-回测
                        if ptrade_trade_cal(trade_date_count) == ptrade_trade_cal(0):
                            stock_history_today = get_history(1, '1d', 'close', stock_code, fq='pre', include=True)
                        else:
                            stock_history_today = get_price(stock_code, start_date=ptrade_trade_cal(trade_date_count), end_date=ptrade_trade_cal(trade_date_count), frequency='1d', fields='close', fq='pre', count=None)
                        # print(stock_code, '今日收盘价:', stock_history_today)
                        # 判断证券今日的最新价是否为空
                        if stock_history_today.empty:
                            # 设定证券今日的最新价缺省值
                            stock_current_price = 0
                        else:
                            # 获取证券今日的最新价
                            stock_current_price = stock_history_today['close'][-1]
                            # print(stock_code, '今日的收盘价:', stock_current_price)

                    # 计算证券今日的MA20
                    stock_days = 20
                    stock_ma20 = round(get_ma_day(stock_days, stock_close_array[-19:], stock_current_price),2)
                    # print(stock_code, '今日的MA20:', stock_ma20)

                # 计算证券今日的最新价突破MA20的数量
                if stock_current_price > stock_ma20:
                    index_member_stock_valid = index_member_stock_valid + 1
                    # print(jingni_index_code, '指数有效股票数量:', index_member_stock_valid)
                elif stock_current_price <= stock_ma20:
                    index_member_stock_invalid = index_member_stock_invalid + 1
                    # print(jingni_index_code, '指数无效股票数量:', index_member_stock_invalid)

            # 计算指数今日趋势得分
            jingni_qsdf_factor_today = round(((index_member_stock_valid/index_member_stock_sum)*100),0)
            # print(jingni_index_code, '指数今日趋势得分: ', jingni_qsdf_factor_today)

            # 获取系统当前日期
            current_datetime = datetime.datetime.strptime(ptrade_trade_cal(trade_date_count), "%Y%m%d").strftime("%Y-%m-%d")
            # print(current_datetime)

            # 检查是否已存在相同数据，如果存在则更新，否则添加
            existing_index = g.jingni_factor_dataframe[(g.jingni_factor_dataframe['日期'] == current_datetime) & (g.jingni_factor_dataframe['指数代码'] == jingni_index_code)]
            if not existing_index.empty:
                # 更新已存在数据
                g.jingni_factor_dataframe.loc[existing_index.index, '趋势得分'] = jingni_qsdf_factor_today
            else:
                # 添加新数据
                g.jingni_factor_dataframe = pandas.concat([g.jingni_factor_dataframe, pandas.DataFrame({'日期': [current_datetime], '指数代码': [jingni_index_code], '趋势得分': [jingni_qsdf_factor_today]})], ignore_index=True)

            # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), jingni_index_code, '趋势得分', jingni_qsdf_factor_today)

        print('因子数据：', g.jingni_factor_dataframe)

    # 如果最新交易时间大于策略结束执行时间
    elif trade_now_time > g.trade_end_time:

        log.info("趋势得分因子休眠时间：%s" % (trade_now_time))

# 计算趋势强度因子数据
def jingni_qsqd_factor(index_list, trade_date_count):

    # # 每次执行创建一个空的DataFrame
    # jingni_qsqd_factor_df = pandas.DataFrame(columns=['日期', '指数代码', '趋势强度'])

    # 获取指数列表
    jingni_index_list = index_list

    for jingni_index_code in jingni_index_list:

        # if jingni_index_code[0] == '8':
            # # 获取申万指数今日行情数据
            # index_current_price_data = tushare_sw_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count), ptrade_trade_cal(trade_date_count))
            # # print(index_current_price_data)
        # else:
            # # 获取中证指数今日行情数据
            # index_current_price_data = tushare_csi_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count), ptrade_trade_cal(trade_date_count))
            # # print(index_current_price_data)

        # # 获取中证&申万指数今日行情数据
        # index_current_price_data = ifind_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count), ptrade_trade_cal(trade_date_count))
        # # print(index_current_price_data)

        # 定义MySQL数据库读取语句
        index_current_price_data_sql = "SELECT 收盘价 FROM jingni_index_daily WHERE 日期 = %s AND 指数代码 = %s"
        # 定义MySQL数据库读取参数
        index_current_price_data_values = (ptrade_trade_cal(trade_date_count), jingni_index_code)
        # 执行MySQL数据库读取函数
        index_current_price_data = pymysql_sql_values(index_current_price_data_sql, index_current_price_data_values)

        if not index_current_price_data:
            # 处理索引超出范围的情况
            index_current_price = 0
            # print(jingni_index_code, '今日的收盘价:', index_current_price)
        else:
            # 将获取的指数今日收盘价数据转换成DataFrame
            index_current_price_data = pandas.DataFrame(index_current_price_data)
            index_current_price = index_current_price_data['收盘价'][0]
            # print(jingni_index_code, '今日的收盘价:', index_current_price)

        # if jingni_index_code[0] == '8':
            # # 获取申万指数最近119日收盘价
            # index_history = tushare_sw_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count-119), ptrade_trade_cal(trade_date_count-1))
            # # print(index_history)
        # else:
            # # 获取中证指数最近119日收盘价
            # index_history = tushare_csi_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count-119), ptrade_trade_cal(trade_date_count-1))
            # # print(index_history)

        # # 获取中证&申万指数最近119日收盘价
        # index_history = ifind_index_daily(jingni_index_code, ptrade_trade_cal(trade_date_count-119), ptrade_trade_cal(trade_date_count-1))
        # # print(index_history)

        # 定义MySQL数据库读取语句
        index_history_sql = "SELECT 收盘价 FROM jingni_index_daily WHERE 日期 >= %s AND 日期 <= %s AND 指数代码 = %s ORDER BY 日期 DESC"
        # 定义MySQL数据库读取参数
        index_history_values = (ptrade_trade_cal(trade_date_count-119), ptrade_trade_cal(trade_date_count-1),jingni_index_code)
        # 执行MySQL数据库读取函数
        index_history = pymysql_sql_values(index_history_sql, index_history_values)

        # 判断证券最近119日的收盘价是否非空
        if not index_history:
            index_history_closet_20day = 0
            index_ma20 = 0
            index_history_closet_60day = 0
            index_ma60 = 0
            index_history_closet_120day = 0
            index_ma120 = 0
        else:
            # 将获取的指数最近119日收盘价数据转换成DataFrame
            index_history = pandas.DataFrame(index_history)
            # 定义证券最近119日的收盘价数组
            index_close_array = index_history['收盘价'].values
            # print(index_close_array)

            if len(index_history['收盘价']) > 18:
                # 获取证券历史第20天的收盘价
                index_history_closet_20day = index_history['收盘价'][18]
                # print(jingni_index_code, '历史第20天的收盘价:', index_history_closet_20day)
                # 计算证券今日的MA20
                index_days = 20
                index_ma20 = round(get_ma_day(index_days, index_close_array[:19], index_current_price),2)
                # print(jingni_index_code, '今日的MA20:', index_ma20)
            else:
                index_history_closet_20day = 0
                index_ma20 = 0  

            if len(index_history['收盘价']) > 58:
                # 获取证券历史第60天的收盘价
                index_history_closet_60day = index_history['收盘价'][58]
                # print(jingni_index_code, '历史第60天的收盘价:', index_history_closet_60day)
                # 计算证券今日的MA60
                index_days = 60
                index_ma60 = round(get_ma_day(index_days, index_close_array[:59], index_current_price),2)
                # print(jingni_index_code, '今日的MA60:', index_ma60)
            else:
                index_history_closet_60day = 0
                index_ma60 = 0  

            if len(index_history['收盘价']) > 118:
                # 获取证券历史第120天的收盘价
                index_history_closet_120day = index_history['收盘价'][118]
                # print(jingni_index_code, '历史第120天的收盘价:', index_history_closet_120day)
                # 计算证券今日的MA120
                index_days = 120
                index_ma120 = round(get_ma_day(index_days, index_close_array[:119], index_current_price),2)
                # print(jingni_index_code, '今日的MA120:', index_ma120)
            else:
                index_history_closet_120day = 0
                index_ma120 = 0  

        if index_history_closet_20day != 0 and index_ma20 != 0 and index_history_closet_60day != 0 and index_ma60 != 0 and index_history_closet_120day != 0 and index_ma120 != 0:

            # 如果证券今日的最新价大于MA20，且今日的最新价大于历史第20天的收盘价，且MA20大于MA60，且今日的最新价大于历史第60天的收盘价，且MA60大于MA120，且今日的最新价大于历史第120天的收盘价
            if index_current_price >= index_ma20 and index_current_price >= index_history_closet_20day and index_ma20 >= index_ma60 and index_current_price >= index_history_closet_60day and index_ma60 >= index_ma120 and index_current_price >= index_history_closet_120day:
                jingni_qsqd_factor_today = 100
    
            # 如果证券今日的最新价大于MA20，且今日的最新价大于历史第20天的收盘价，且MA20大于MA60，且今日的最新价大于历史第60天的收盘价，且MA60大于MA120
            elif index_current_price >= index_ma20 and index_current_price >= index_history_closet_20day and index_ma20 >= index_ma60 and index_current_price >= index_history_closet_60day and index_ma60 >= index_ma120:
                jingni_qsqd_factor_today = 100
    
            # 如果证券今日的最新价大于MA20，且今日的最新价大于历史第20天的收盘价，且MA20大于MA60，且今日的最新价大于历史第60天的收盘价
            elif index_current_price >= index_ma20 and index_current_price >= index_history_closet_20day and index_ma20 >= index_ma60 and index_current_price >= index_history_closet_60day:
                jingni_qsqd_factor_today = 80
    
            # 如果证券今日的最新价大于MA20，且今日的最新价大于历史第20天的收盘价，且MA20大于MA60
            elif index_current_price >= index_ma20 and index_current_price >= index_history_closet_20day and index_ma20 >= index_ma60:
                jingni_qsqd_factor_today = 60
    
            # 如果证券今日的最新价大于MA20，且今日的最新价大于历史第20天的收盘价
            elif index_current_price >= index_ma20 and index_current_price >= index_history_closet_20day:
                jingni_qsqd_factor_today = 40
    
            # 如果证券今日的最新价大于MA20
            elif index_current_price >= index_ma20:
                jingni_qsqd_factor_today = 20
    
            # 如果证券今日的最新价小于MA20
            elif index_current_price < index_ma20:
                jingni_qsqd_factor_today = 0

        else:
            jingni_qsqd_factor_today = 0

        # print(jingni_index_code, '指数今日趋势强度: ', jingni_qsqd_factor_today)

        # 系统当前日期
        current_datetime = datetime.datetime.strptime(ptrade_trade_cal(trade_date_count), "%Y%m%d").strftime("%Y-%m-%d")
        # print(current_datetime)

        # 检查是否已存在相同数据，如果存在则更新，否则添加
        existing_index = g.jingni_factor_dataframe[(g.jingni_factor_dataframe['日期'] == current_datetime) & (g.jingni_factor_dataframe['指数代码'] == jingni_index_code)]
        if not existing_index.empty:
            # 更新已存在数据
            g.jingni_factor_dataframe.loc[existing_index.index, '趋势强度'] = jingni_qsqd_factor_today
        else:
            # 添加新数据
            g.jingni_factor_dataframe = pandas.concat([g.jingni_factor_dataframe, pandas.DataFrame({'日期': [current_datetime], '指数代码': [jingni_index_code], '趋势强度': [jingni_qsqd_factor_today]})], ignore_index=True)

        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), jingni_index_code, '趋势强度', jingni_qsqd_factor_today)

    print('因子数据：', g.jingni_factor_dataframe)

# 计算两融资金因子数据
def jingni_lrzj_factor(index_list, trade_date_count):

    # 定义MySQL数据库读取语句
    stocks_margin_detail_list_onedayago_sql = "SELECT ts_code, rzmre, rzche FROM margin_detail WHERE trade_date = %s"
    # 定义MySQL数据库读取参数
    stocks_margin_detail_list_onedayago_values = (ptrade_trade_cal(trade_date_count))
    # print(stocks_margin_detail_list_onedayago_values)
    # 执行MySQL数据库读取函数
    stocks_margin_detail_list_onedayago = pymysql_sql_values(stocks_margin_detail_list_onedayago_sql, stocks_margin_detail_list_onedayago_values)
    #print('获取T-1交易日融资融券交易明细：%s' % (stocks_margin_detail_list_onedayago))

    # 判断MySQL数据库查询结果是否不为空
    if stocks_margin_detail_list_onedayago:
        stocks_margin_detail_list_onedayago_None = False
        stocks_margin_detail_list_onedayago = pandas.DataFrame(stocks_margin_detail_list_onedayago)
        # print('获取T-1交易日融资融券交易明细：%s' % (stocks_margin_detail_list_onedayago))
    else:
        stocks_margin_detail_list_onedayago_None = True

    # 获取策略计算的指数列表
    jingni_index_list = index_list
    #print('获取策略计算的指数列表：%s' % (index_list))

    # # 每次执行创建一个空的DataFrame
    # jingni_lrzj_factor_df = pandas.DataFrame(columns=['日期', '指数代码', '两融资金净买入'])

    for jingni_index_code in jingni_index_list:

        # 计算指数的两融资金净买入额
        index_margin_detail_money = 0

        # 判断是否不为空
        if not stocks_margin_detail_list_onedayago_None:

            # 获取指数成分列表 
            stock_code_list = g.jingni_index_security_code_list[g.jingni_index_security_code_list['指数代码'] == jingni_index_code]['股票代码'].tolist()
            # print(jingni_index_code,'指数成分列表:')

            for stock_code in stock_code_list:

                # 检查'rzmre'列是否不包含NaN值
                if not stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzmre'].isnull().any():
                    # 获取股票T-1交易日沪深融资买入额
                    if len(stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzmre']) != 0:
                        stocks_margin_detail_onedayago = stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzmre'].iloc[0]
                        # print(stock_code, '股票T-1日的沪深融资买入额:', stocks_margin_detail_onedayago)
                    elif len(stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzmre']) == 0:
                        stocks_margin_detail_onedayago = 0
                        # print(stock_code, '股票T-1日的沪深融资买入额:', stocks_margin_detail_onedayago)
                else:
                    stocks_margin_detail_onedayago = 0
                    # print(stock_code, '股票T-1日的沪深融资买入额:', stocks_margin_detail_onedayago)
    
                # 检查'rzche'列是否不包含NaN值
                if not stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzche'].isnull().any():
                    # 获取股票T-1交易日沪深融资偿还额
                    if len(stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzche']) != 0:
                        stocks_margin_detail_twodayago = stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzche'].iloc[0]
                        # print(stock_code, '股票T-1日的沪深融资偿还额:', stocks_margin_detail_twodayago)
                    elif len(stocks_margin_detail_list_onedayago.loc[stocks_margin_detail_list_onedayago['ts_code'] == stock_code, 'rzche']) == 0:
                        stocks_margin_detail_twodayago = 0
                        # print(stock_code, '股票T-1日的沪深融资偿还额:', stocks_margin_detail_twodayago)
                else:
                    stocks_margin_detail_twodayago = 0
                    # print(stock_code, '股票T-1日的沪深融资偿还额:', stocks_margin_detail_twodayago)

                # 计算股票的两融资金净买入额
                stocks_margin_detail_money = stocks_margin_detail_onedayago - stocks_margin_detail_twodayago
                # print(stock_code, '股票的两融资金净买入额:', stocks_margin_detail_money)
                index_margin_detail_money = index_margin_detail_money + stocks_margin_detail_money
                # print(stock_code, '累加的两融资金净买入额:', index_margin_detail_money)

        # 计算指数T-1交易日两融资金净买入
        jingni_lrzj_score_yesterday = float(round(index_margin_detail_money/100000000, 2))
        # print(jingni_index_code, '指数', trade_date, '两融资金净买入: ', jingni_lrzj_score_yesterday, '亿元')

        # 系统当前日期
        current_datetime = datetime.datetime.strptime(ptrade_trade_cal(trade_date_count), "%Y%m%d").strftime("%Y-%m-%d")

        # 检查是否已存在相同数据，如果存在则更新，否则添加
        existing_index = g.jingni_factor_dataframe[(g.jingni_factor_dataframe['日期'] == current_datetime) & (g.jingni_factor_dataframe['指数代码'] == jingni_index_code)]
        if not existing_index.empty:
            # 更新已存在数据
            g.jingni_factor_dataframe.loc[existing_index.index, '两融资金净买入'] = jingni_lrzj_score_yesterday
        else:
            # 添加新数据
            g.jingni_factor_dataframe = pandas.concat([g.jingni_factor_dataframe, pandas.DataFrame({'日期': [current_datetime], '指数代码': [jingni_index_code], '两融资金净买入': [jingni_lrzj_score_yesterday]})], ignore_index=True)

        # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), jingni_index_code, '两融资金净买入', jingni_lrzj_score_yesterday)

    print('因子数据：', g.jingni_factor_dataframe)

# 因子归一化处理
def normalize(factor_values, feature_range=(-0.5, 0.5), verbose=False):
    """
    优化后的归一化函数，支持自定义输出范围和处理负数
    
    参数:
        factor_values : array-like 
            待归一化的数据（支持numpy数组、pandas Series等）
        feature_range : tuple, optional 
            目标范围，默认为(-0.5, 0.5)
        verbose : bool, optional
            是否打印过程信息
    
    返回:
        normalized : 与输入类型一致的归一化后数组
    """
    # 转换为numpy数组以统一处理
    values = numpy.asarray(factor_values)
    
    min_val = values.min()
    max_val = values.max()
    
    if verbose:
        print('原始范围:', min_val, max_val)
    
    # 处理全相同值的情况
    if numpy.isclose(min_val, max_val):
        normalized = numpy.full_like(values, feature_range[0], dtype=numpy.float64)
    else:
        # 线性归一化到目标范围
        normalized = (values - min_val) / (max_val - min_val)
        # 缩放至自定义区间（如[-1,1]）
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    
    if verbose:
        print('目标范围:', feature_range)
        print('归一化结果前五项:', normalized[:5])
    
    # 保持与输入一致的数据类型（如pandas Series）
    if isinstance(factor_values, (pandas.Series, pandas.DataFrame)):
        return type(factor_values)(normalized, index=factor_values.index)

    return normalized

# 计算多因子分数
def jingni_index_multi_factor_selection(weights, index_multi_factor_data):

    if index_multi_factor_data is not None and not index_multi_factor_data.empty:

        # 将日期列转换为datetime类型
        index_multi_factor_data['日期'] = pandas.to_datetime(index_multi_factor_data['日期'])

        # for index_code, group in  index_multi_factor_data.groupby('指数代码'):
        #     # 将今天趋势得分的值更新为今天趋势得分减去昨天趋势得分的差值
        #     index_multi_factor_data.loc[( index_multi_factor_data['指数代码'] == index_code) & ( index_multi_factor_data['日期'] == ptrade_trade_cal(0)), '趋势得分'] = group.loc[group['日期'] == ptrade_trade_cal(0), '趋势得分'].values[0] - group.loc[group['日期'] == ptrade_trade_cal(-1), '趋势得分'].values[0]
        #     # 将今天趋势强度的值更新为昨天趋势强度的值
        #     index_multi_factor_data.loc[( index_multi_factor_data['指数代码'] == index_code) & ( index_multi_factor_data['日期'] == ptrade_trade_cal(0)), '趋势强度'] = group.loc[group['日期'] == ptrade_trade_cal(-1), '趋势强度'].values[0]
        #     # 将今天两融资金净买入的值更新为昨天两融资金净买入的值
        #     index_multi_factor_data.loc[( index_multi_factor_data['指数代码'] == index_code) & ( index_multi_factor_data['日期'] == ptrade_trade_cal(0)), '两融资金净买入'] = group.loc[group['日期'] == ptrade_trade_cal(-1), '两融资金净买入'].values[0]
        # print('预处理因子数据：', index_multi_factor_data)

        # 筛选出日期等于今天的数据
        index_multi_factor_data = index_multi_factor_data.loc[index_multi_factor_data['日期'] == ptrade_trade_cal(0)]
        print('今天预处理因子数据：', index_multi_factor_data)

        # 迭代weights字典中的每个因子，进行归一化处理并计算多因子分数
        for factor, weight in weights.items():

            # 因子归一化处理
            index_multi_factor_data[factor] = normalize(index_multi_factor_data[factor], feature_range=(-0.5, 0.5), verbose=False)

            # 如果index_multi_factor_data没有'多因子分数'字段则创建并赋值
            if '多因子分数' not in index_multi_factor_data:
                index_multi_factor_data['多因子分数'] = index_multi_factor_data[factor] * weight
                # print(index_multi_factor_data['多因子分数'])

            # 如果index_multi_factor_data已有'多因子分数'字段则累加多个因子权重分数
            else:
                index_multi_factor_data['多因子分数'] += index_multi_factor_data[factor] * weight
                # print(index_multi_factor_data['多因子分数'])

        # 多因子分数降序
        index_multi_factor_data = index_multi_factor_data.sort_values(by='多因子分数', ascending=False)
        # print('多因子分数降序：',index_multi_factor_data)

        return index_multi_factor_data

# 计算择时信号值
def jingni_index_trading_signal_selection(index_trading_signal_data):

    if index_trading_signal_data is not None and not index_trading_signal_data.empty:

        # 将index_trading_signal_data转换成DataFrame
        index_trading_signal_data = pandas.DataFrame(index_trading_signal_data)
        # 将index_trading_signal_data的日期字符串类型转换成日期时间类型
        index_trading_signal_data['日期'] = pandas.to_datetime(index_trading_signal_data['日期'], format='%Y-%m-%d')
        # print(index_trading_signal_data)

        # 将index_trading_signal_data的日期格式为YYYY-MM-DD，将ptrade_trade_cal函数获取的日期格式从YYYYMMDD转换成YYYY-MM-DD
        date_today = datetime.datetime.strptime(ptrade_trade_cal(0), "%Y%m%d").strftime("%Y-%m-%d")

        # 将index_trading_signal_data中日期等于今天的指数代码筛选出来
        jingni_index_code_list = index_trading_signal_data[index_trading_signal_data['日期'] == date_today]['指数代码']

        for index_code in jingni_index_code_list:

            # 获取当前交易日指数趋势得分字段的最新值
            jingni_qsdf_factor_score_today = index_trading_signal_data[index_trading_signal_data['指数代码'] == index_code].sort_values(by='日期', ascending=False).iloc[0]['趋势得分']
            # print("%s指数今日趋势得分：%s" % (index_code, jingni_qsdf_factor_score_today))
            # 获取前一交易日指数趋势得分字段的历史值
            jingni_qsdf_factor_score_yesterday = index_trading_signal_data[index_trading_signal_data['指数代码'] == index_code].sort_values(by='日期', ascending=False).iloc[1]['趋势得分']
            # print("%s指数昨日趋势得分：%s" % (index_code, jingni_qsdf_factor_score_yesterday))
            # 获取前五交易日指数趋势得分字段的最小值
            jingni_qsdf_factor_score_min = index_trading_signal_data[index_trading_signal_data['指数代码'] == index_code].sort_values(by='日期', ascending=False)['趋势得分'].dropna().iloc[1:6].min()
            # print("%s指数最低趋势得分：%s" % (index_code, jingni_qsdf_factor_score_min))
            # 获取前五交易日指数趋势得分字段的最大值
            jingni_qsdf_factor_score_max = index_trading_signal_data[index_trading_signal_data['指数代码'] == index_code].sort_values(by='日期', ascending=False)['趋势得分'].dropna().iloc[1:6].max()
            # print("%s指数最高趋势得分：%s" % (index_code, jingni_qsdf_factor_score_max))

            # 为index_trading_signal_data新增择时信号值字段，并通过条件判断设置择时信号值的数值
            # if jingni_qsdf_factor_score_today > jingni_qsdf_factor_score_yesterday and jingni_qsdf_factor_score_today > jingni_qsdf_factor_score_max:
            if jingni_qsdf_factor_score_today > jingni_qsdf_factor_score_max:
                index_trading_signal_data.loc[(index_trading_signal_data['日期'] == date_today) & (index_trading_signal_data['指数代码'] == index_code), '择时信号值'] = 0.5
            # elif jingni_qsdf_factor_score_today > jingni_qsdf_factor_score_yesterday and jingni_qsdf_factor_score_today > jingni_qsdf_factor_score_min:
            elif jingni_qsdf_factor_score_max >= jingni_qsdf_factor_score_today >= jingni_qsdf_factor_score_min:
                index_trading_signal_data.loc[(index_trading_signal_data['日期'] == date_today) & (index_trading_signal_data['指数代码'] == index_code), '择时信号值'] = 0
            # elif jingni_qsdf_factor_score_today <= jingni_qsdf_factor_score_yesterday:
            elif jingni_qsdf_factor_score_min > jingni_qsdf_factor_score_today:
                index_trading_signal_data.loc[(index_trading_signal_data['日期'] == date_today) & (index_trading_signal_data['指数代码'] == index_code), '择时信号值'] = -0.5

            index_trading_signal_score = index_trading_signal_data.loc[(index_trading_signal_data['日期'] == date_today) & (index_trading_signal_data['指数代码'] == index_code)]
            # print("%s指数择时信号值：%s" % (index_code, index_trading_signal_score.iloc[0]['择时信号值']))

        # 将index_trading_signal_data日期等于今天的记录筛选出来
        index_trading_signal_data = index_trading_signal_data[index_trading_signal_data['日期'] == datetime.datetime.strptime(ptrade_trade_cal(0), "%Y%m%d")]
        # print(index_trading_signal_data)

        # 择时信号值降序
        index_trading_signal_data = index_trading_signal_data.sort_values(by='择时信号值', ascending=False)
        # print('择时信号值降序：',index_trading_signal_data)

        return index_trading_signal_data

# 计算EMA信号平滑
def calc_ema_signal(signal_series, span=5):

    if len(signal_series) < span:
        return float(signal_series.iloc[-1]) if len(signal_series) > 0 else 0.0

    ema = signal_series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1])

# 卡尔曼滤波平滑
def kalman_filter_smooth(observations, process_noise=1e-4, measurement_noise=1e-2):

    n = len(observations)
    if n == 0:
        return 0.0

    x_est = float(observations[0])
    p_est = 1.0
    filtered = [x_est]

    for k in range(1, n):
        x_pred = x_est
        p_pred = p_est + process_noise

        if measurement_noise > 0:
            kg = p_pred / (p_pred + measurement_noise)
        else:
            kg = 1.0

        z = float(observations[k])
        x_est = x_pred + kg * (z - x_pred)
        p_est = (1.0 - kg) * p_pred
        filtered.append(x_est)

    return filtered[-1]

# 计算滚动波动率
def calc_rolling_volatility(security_code, lookback=20):

    try:
        history = get_history(lookback + 1, '1d', 'close', security_code, fq='pre', include=False)
        if history is None or history.empty or len(history) < 5:
            return 0.25

        returns = history['close'].pct_change().dropna()
        if len(returns) < 3:
            return 0.25

        daily_vol = float(numpy.std(returns.values))
        annual_vol = daily_vol * numpy.sqrt(252)
        return max(annual_vol, 0.05)
    except:
        return 0.25

# 波动率自适应仓位因子
def calc_volatility_adaptive_factor(target_vol=0.15, current_vol=None):

    if current_vol is None or current_vol <= 0:
        return 0.5

    factor = target_vol / current_vol
    return max(0.25, min(factor, 1.0))

# 凯利公式仓位管理
def calc_kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, max_fraction=0.40):

    if win_rate <= 0 or win_loss_ratio <= 0:
        return 0.0

    b = win_loss_ratio
    p = win_rate
    q = 1.0 - p

    kelly = (p * b - q) / b

    if kelly <= 0:
        return 0.0

    half_kelly = kelly * 0.5
    return min(half_kelly, max_fraction)

# 因子对称正交化处理
def factor_symmetric_orthogonal(factor_values_df, factor_cols):

    if factor_values_df is None or factor_values_df.empty:
        return factor_values_df

    valid_df = factor_values_df[factor_cols].dropna()
    if valid_df.empty or len(valid_df) < len(factor_cols):
        return factor_values_df

    try:
        values = valid_df.values.astype(numpy.float64)
        values_std = (values - numpy.mean(values, axis=0)) / (numpy.std(values, axis=0) + 1e-10)

        corr_matrix = numpy.corrcoef(values_std.T)
        eigenvalues, eigenvectors = numpy.linalg.eigh(corr_matrix)

        eigenvalues = numpy.maximum(eigenvalues, 1e-8)
        sqrt_inv_eigenvalues = numpy.diag(1.0 / numpy.sqrt(eigenvalues))

        orthogonal_matrix = eigenvectors @ sqrt_inv_eigenvalues @ eigenvectors.T

        orthogonal_values = values_std @ orthogonal_matrix

        result_df = factor_values_df.copy()
        for i, col in enumerate(factor_cols):
            if col in result_df.columns and i < orthogonal_values.shape[1]:
                result_df[col] = orthogonal_values[:, i]

        return result_df
    except:
        return factor_values_df

# 计算因子IC动态权重
def calc_factor_ic_weights(factor_values_df, factor_cols, lookback=30):

    default_weights = {col: 1.0/len(factor_cols) for col in factor_cols}

    if factor_values_df is None or factor_values_df.empty:
        return default_weights

    try:
        dates_sorted = sorted(factor_values_df['日期'].unique())
        if len(dates_sorted) < lookback:
            return default_weights

        latest_date = dates_sorted[-1]
        lookback_start = max(0, len(dates_sorted) - lookback - 1)
        lookback_dates = dates_sorted[lookback_start:]

        ic_scores = {col: [] for col in factor_cols}

        for i in range(1, len(lookback_dates)):
            prev_date = lookback_dates[i-1]
            curr_date = lookback_dates[i]

            prev_data = factor_values_df[factor_values_df['日期'] == prev_date]
            curr_data = factor_values_df[factor_values_df['日期'] == curr_date]

            if prev_data.empty or curr_data.empty:
                continue

            merged = pandas.merge(
                prev_data[['指数代码'] + factor_cols],
                curr_data[['指数代码']],
                on='指数代码',
                how='inner'
            )

            if len(merged) < 5:
                continue

            for col in factor_cols:
                prev_rank = merged[col].rank(pct=True)
                curr_rank = merged[col].rank(pct=True)
                ic = numpy.corrcoef(prev_rank, curr_rank)[0, 1]
                if not numpy.isnan(ic):
                    ic_scores[col].append(abs(ic))

        ic_weights = {}
        total_ic = 0.0
        for col in factor_cols:
            if ic_scores[col]:
                mean_ic = numpy.mean(ic_scores[col])
            else:
                mean_ic = 0.0
            ic_weights[col] = mean_ic
            total_ic += mean_ic

        if total_ic > 0:
            for col in factor_cols:
                ic_weights[col] = ic_weights[col] / total_ic
        else:
            ic_weights = default_weights

        return ic_weights

    except:
        return default_weights

# 波动率倒数加权
def calc_volatility_inverse_weights(security_codes, lookback=20):

    result = {}
    if not security_codes:
        return result

    vols = {}
    for code in security_codes:
        vol = calc_rolling_volatility(code, lookback)
        vols[code] = vol

    inverse_vols = {code: 1.0/max(vol, 0.05) for code, vol in vols.items()}
    total_inverse = sum(inverse_vols.values())

    if total_inverse > 0:
        for code in security_codes:
            result[code] = inverse_vols[code] / total_inverse
    else:
        n = len(security_codes)
        for code in security_codes:
            result[code] = 1.0 / n

    return result

# 组合止损检查
def check_portfolio_stop_loss():

    if not hasattr(g, 'peak_value') or g.peak_value == 0:
        g.peak_value = g.assets_value
        g.stop_loss_triggered = False
        return False

    if g.assets_value > g.peak_value:
        g.peak_value = g.assets_value

    drawdown = (g.peak_value - g.assets_value) / g.peak_value * 100

    if not hasattr(g, 'stop_loss_triggered'):
        g.stop_loss_triggered = False

    if drawdown > 15.0:
        log.info("触发组合15%%强制止损，当前回撤: %.2f%%" % drawdown)
        g.stop_loss_triggered = True
        return True
    elif drawdown > 8.0:
        log.info("触发组合8%%减仓止损，当前回撤: %.2f%%" % drawdown)
        g.stop_loss_triggered = True
        return True
    else:
        g.stop_loss_triggered = False
        return False

# 单标的止损检查
def check_single_position_stop_loss(stop_loss_pct=0.10):

    positions = get_position()
    stocks_to_sell = []

    for stock_code in positions:
        pos = positions[stock_code]
        if pos.amount > 0 and pos.cost_basis > 0:
            current_price = pos.last_sale_price
            loss_pct = (current_price - float(pos.cost_basis)) / float(pos.cost_basis)
            if loss_pct < -stop_loss_pct:
                stocks_to_sell.append(stock_code)
                log.info("%s 触发单标止损，亏损: %.2f%%" % (stock_code, loss_pct * 100))

    return stocks_to_sell

# 获取择时信号指数代码（930903.CSI 趋势得分）
def get_aggregate_timing_signal():

    if g.jingni_factor_dataframe is None or g.jingni_factor_dataframe.empty:
        return pandas.Series(dtype=float)

    try:
        today_data = g.jingni_factor_dataframe[
            g.jingni_factor_dataframe['日期'] == datetime.datetime.strptime(ptrade_trade_cal(0), "%Y%m%d").strftime("%Y-%m-%d")
        ]
    except:
        try:
            today_data = g.jingni_factor_dataframe[
                g.jingni_factor_dataframe['日期'] == ptrade_trade_cal(0)
            ]
        except:
            today_data = g.jingni_factor_dataframe

    if today_data.empty:
        return pandas.Series(dtype=float)

    if '趋势得分' in today_data.columns:
        return today_data['趋势得分'].dropna()

    return pandas.Series(dtype=float)

# 策略分析决策函数
def trading_strategy(context):

    log.info('开始量化交易策略')

    # 获取当日前交易账户信息
    portfolio(context)

    # 检查组合止损
    is_stop_loss = check_portfolio_stop_loss()
    drawdown_pct = 0.0
    if hasattr(g, 'peak_value') and g.peak_value > 0:
        drawdown_pct = (g.peak_value - g.assets_value) / g.peak_value * 100

    # 计算因子数据
    jingni_qsdf_factor(g.jingni_index_code_list, 0)
    jingni_qsqd_factor(g.jingni_index_code_list, 0)
    jingni_lrzj_factor(g.jingni_index_code_list, 0)

    factor_cols = ['趋势得分', '趋势强度', '两融资金净买入']
    default_weights = {'趋势得分': 0.5, '趋势强度': 0.3, '两融资金净买入': 0.2}

    # 因子正交化处理
    factor_data_processed = factor_symmetric_orthogonal(g.jingni_factor_dataframe, factor_cols)

    # IC动态权重计算
    ic_weights = calc_factor_ic_weights(g.jingni_factor_dataframe, factor_cols, lookback=30)
    if ic_weights and sum(ic_weights.values()) > 0:
        active_weights = ic_weights
        log.info("IC动态权重: %s" % active_weights)
    else:
        active_weights = default_weights
        log.info("使用默认权重: %s" % active_weights)

    # 多因子选股
    multi_factor_result = jingni_index_multi_factor_selection(active_weights, factor_data_processed)

    log.info("多因子排名结果: %s" % multi_factor_result)

    if multi_factor_result is not None and not multi_factor_result.empty:
        top_n = min(8, len(multi_factor_result))
        stocks_entrust_list = multi_factor_result.head(top_n)['指数代码'].tolist()
        stocks_entrust_list = [securities_code_conversion(code) for code in stocks_entrust_list]
        log.info("因子选股(%s只): %s" % (top_n, stocks_entrust_list))

        # 波动率倒数加权
        vol_inv_weights = calc_volatility_inverse_weights(stocks_entrust_list, lookback=20)
        security_weight_map = vol_inv_weights
        log.info("波动率倒数权重: %s" % security_weight_map)
    else:
        stocks_entrust_list = []
        security_weight_map = {}

    # 获取当前持仓
    market_position = get_position()
    stocks_positions_list = []
    position_cost_map = {}
    for stock_code in market_position:
        pos_code = securities_code_conversion(stock_code)
        stocks_positions_list.append(pos_code)
        pos_obj = market_position[stock_code]
        if pos_obj.amount > 0 and float(pos_obj.cost_basis) > 0:
            position_cost_map[pos_code] = {
                'amount': pos_obj.amount,
                'cost_basis': float(pos_obj.cost_basis),
                'last_price': float(pos_obj.last_sale_price)
            }
    log.info("当前持仓: %s" % stocks_positions_list)

    # 获取择时信号 — EMA平滑 + 卡尔曼滤波
    timing_signal_raw = get_aggregate_timing_signal()

    timing_signal_smoothed = None
    timing_decision = 'hold'

    if timing_signal_raw is not None and len(timing_signal_raw) > 0:
        signal_series = pandas.Series(timing_signal_raw.values, dtype=float)

        # EMA平滑
        ema_signal = calc_ema_signal(signal_series, span=5)
        log.info("EMA平滑择时信号: %.2f" % ema_signal)

        # 卡尔曼滤波平滑
        kalman_input = signal_series.values
        kalman_signal = kalman_filter_smooth(kalman_input, process_noise=5e-4, measurement_noise=1e-2)
        timing_signal_smoothed = kalman_signal
        log.info("卡尔曼滤波择时信号: %.2f" % kalman_signal)

        # 信号判断：EMA趋势向上+卡尔曼滤波正值 → 买入；EMA趋势向下+卡尔曼滤波负值 → 卖出
        if len(signal_series) >= 3:
            ema_prev = calc_ema_signal(signal_series.iloc[:-1], span=5)
            if ema_signal > ema_prev and kalman_signal >= 0:
                timing_decision = 'buy'
            elif ema_signal < ema_prev and kalman_signal < 0:
                timing_decision = 'sell'
            elif kalman_signal >= 0:
                timing_decision = 'buy'
            else:
                timing_decision = 'sell'
        else:
            timing_decision = 'hold'
    else:
        timing_decision = 'hold'

    log.info("择时决策: %s (is_stop_loss=%s, drawdown=%.2f%%)" % (timing_decision, is_stop_loss, drawdown_pct))

    # 组合止损处理
    if is_stop_loss:
        if drawdown_pct > 15.0:
            for stock_code in stocks_positions_list:
                stock_enable_amount = get_position(stock_code).enable_amount
                stock_last_sale_price = get_position(stock_code).last_sale_price
                stock_value = -stock_enable_amount * stock_last_sale_price
                stock_order_value(context, stock_code, stock_value)
            g.order_sell_flag = True
            log.info("组合强制清仓完成")

            send_msg = '%s：\n触发强制清仓止损！当前回撤: %.2f%%' % (g.__strategy_name, drawdown_pct)
            try:
                WechatNotifier("ww137c94e7a2caf6fb", "AnFfd7UKYGHEq8hpmZvHI_sOHDDvkcAxCcbE6T09Z_I", "1000003").send_qywx_text(send_msg, '', 'duhanjun')
            except:
                pass
            log.info('结束量化交易策略（强制清仓）')
            return

        elif drawdown_pct > 8.0:
            if not g.order_sell_flag and len(stocks_positions_list) > 0:
                for stock_code in stocks_positions_list:
                    stock_enable_amount = get_position(stock_code).enable_amount
                    stock_last_sale_price = get_position(stock_code).last_sale_price
                    stock_value = -stock_enable_amount * stock_last_sale_price * 0.5
                    stock_order_value(context, stock_code, stock_value)
                g.order_sell_flag = True
                log.info("组合减半仓止损完成")
            log.info('结束量化交易策略（减半仓止损）')
            return

    # 单标的止损
    single_stop_stocks = check_single_position_stop_loss(stop_loss_pct=0.10)
    if single_stop_stocks:
        for stock_code in single_stop_stocks:
            stock_enable_amount = get_position(stock_code).enable_amount
            stock_last_sale_price = get_position(stock_code).last_sale_price
            stock_value = -stock_enable_amount * stock_last_sale_price
            stock_order_value(context, stock_code, stock_value)
        log.info("单标止损完成: %s" % single_stop_stocks)

    # 信号驱动交易决策
    if timing_decision == 'sell':
        if not g.order_sell_flag and g.assets_market_ratio > 0:
            for stock_code in stocks_positions_list:
                stock_enable_amount = get_position(stock_code).enable_amount
                stock_last_sale_price = get_position(stock_code).last_sale_price
                stock_value = -stock_enable_amount * stock_last_sale_price
                stock_order_value(context, stock_code, stock_value)
            ## g.order_sell_flag = True
            log.info("信号驱动卖出完成")

    elif timing_decision == 'buy' and len(stocks_entrust_list) > 0:
        # 卖出与回购解耦 — 回购仅在有实际卖出后的回弹信号触发
        if not g.order_buyback_flag:
            trades_info = get_trades()
            trading_total_money = 0
            for trade_list in trades_info.values():
                for trade in trade_list:
                    if len(trade) > 6:
                        if trade[3] == '卖':
                            trading_total_money = trading_total_money + float(trade[6])
            if trading_total_money > 0 and kalman_filter_smooth(timing_signal_raw.values, 5e-4, 1e-2) >= 0:
                buyback_weight = round(trading_total_money / g.assets_value, 4)
                log.info("策略回购仓位比例: %.2f%%" % (buyback_weight * 100))
                for stock_code in stocks_entrust_list:
                    sw = security_weight_map.get(stock_code, 1.0/max(1, len(stocks_entrust_list)))
                    stock_value = g.assets_value * sw * buyback_weight
                    stock_order_value(context, stock_code, stock_value)
                g.order_buyback_flag = True
                log.info("回购完成，回购标识: True")

        # 主买入逻辑
        if not g.order_buy_flag and g.assets_market_ratio <= 85.00:
            # 波动率自适应因子
            benchmark_code = '000906.SS'
            try:
                benchmark_vol = calc_rolling_volatility(benchmark_code, lookback=20)
            except:
                benchmark_vol = 0.25
            vol_adaptive_factor = calc_volatility_adaptive_factor(target_vol=0.15, current_vol=benchmark_vol)
            log.info("波动率自适应因子: %.3f (基准波动率: %.2f%%)" % (vol_adaptive_factor, benchmark_vol * 100))

            # 凯利公式因子
            kelly_factor = calc_kelly_fraction(win_rate=0.55, win_loss_ratio=1.5, max_fraction=0.30)
            log.info("凯利公式因子: %.3f" % kelly_factor)

            # 买入权重 = min(凯利推荐, 基准权重) * 波动率自适应因子
            order_base_weight = 0.20
            final_order_weight = min(kelly_factor, order_base_weight) * vol_adaptive_factor
            final_order_weight = max(0.05, min(final_order_weight, 0.30))
            log.info("最终买入权重: %.3f" % final_order_weight)

            if final_order_weight > 0:
                for stock_code in stocks_entrust_list:
                    sw = security_weight_map.get(stock_code, 1.0/max(1, len(stocks_entrust_list)))
                    stock_value = g.assets_value * sw * final_order_weight
                    stock_order_value(context, stock_code, stock_value)
                g.order_buy_flag = True
                log.info("买入完成，买入标识: True")

    elif timing_decision == 'hold':
        log.info("当前信号为持有，不执行交易")

    log.info('结束量化交易策略')

# 执行策略分析决策
def run_trading_strategy(context):
    trading_strategy(context)

# 盘中计算中证行业指数趋势得分因子
def run_jingni_csi_qsdf_factor(context):
    g.jingni_csi_qsdf_factor_dataframe_today = jingni_qsdf_factor(g.jingni_csi_index_code_list, 0)
    print(g.jingni_csi_qsdf_factor_dataframe_today)

# 盘中计算申万行业指数趋势得分因子
def run_jingni_swi_qsdf_factor(context):
    g.jingni_swi_qsdf_factor_dataframe_today = jingni_qsdf_factor(g.jingni_swi_index_code_list, 0)
    print(g.jingni_swi_qsdf_factor_dataframe_today)

# 盘中计算中证行业指数趋势强度因子
def run_jingni_csi_qsqd_factor(context):
    g.jingni_csi_qsqd_factor_dataframe_today = jingni_qsqd_factor(g.jingni_csi_index_code_list, 0)
    print(g.jingni_csi_qsqd_factor_dataframe_today)

# 盘中计算申万行业指数趋势强度因子
def run_jingni_swi_qsqd_factor(context):
    g.jingni_swi_qsqd_factor_dataframe_today = jingni_qsqd_factor(g.jingni_swi_index_code_list, 0)
    print(g.jingni_swi_qsqd_factor_dataframe_today)

# 盘中计算中证行业指数两融资金净买入因子
def run_jingni_csi_lrzj_factor(context):
    g.jingni_csi_lrzj_factor_dataframe_today = jingni_lrzj_factor(g.jingni_csi_index_code_list, 0)
    print(g.jingni_csi_lrzj_factor_dataframe_today)

# 盘中计算申万行业指数两融资金净买入因子
def run_jingni_swi_lrzj_factor(context):
    g.jingni_swi_lrzj_factor_dataframe_today = jingni_lrzj_factor(g.jingni_swi_index_code_list, 0)
    print(g.jingni_swi_lrzj_factor_dataframe_today)

# 盘中计算中证行业指数多因子分数
def run_jingni_csi_multi_factor_selection(context):

    # 检查因子数据是否为空
    if (
        g.jingni_csi_qsdf_factor_dataframe_today is None or 
        g.jingni_csi_qsdf_factor_dataframe_1d_ago is None or 
        g.jingni_csi_qsdf_factor_dataframe_2d_ago is None or 
        g.jingni_csi_qsdf_factor_dataframe_3d_ago is None or 
        g.jingni_csi_qsdf_factor_dataframe_4d_ago is None
        ):
        index_trading_signal_data = None
    else:
        # 纵向合并所有DataFrame
        index_trading_signal_data = pandas.concat([
            g.jingni_csi_qsdf_factor_dataframe_today[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_1d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_2d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_3d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_4d_ago[['日期', '指数代码', '趋势得分']],

            g.jingni_csi_qsqd_factor_dataframe_today[['日期', '指数代码', '趋势强度']],
            g.jingni_csi_qsqd_factor_dataframe_1d_ago[['日期', '指数代码', '趋势强度']],
            g.jingni_csi_qsqd_factor_dataframe_2d_ago[['日期', '指数代码', '趋势强度']],
            g.jingni_csi_qsqd_factor_dataframe_3d_ago[['日期', '指数代码', '趋势强度']],
            g.jingni_csi_qsqd_factor_dataframe_4d_ago[['日期', '指数代码', '趋势强度']],

            g.jingni_csi_lrzj_factor_dataframe_today[['日期', '指数代码', '两融资金净买入']],
            g.jingni_csi_lrzj_factor_dataframe_1d_ago[['日期', '指数代码', '两融资金净买入']],
            g.jingni_csi_lrzj_factor_dataframe_2d_ago[['日期', '指数代码', '两融资金净买入']],
            g.jingni_csi_lrzj_factor_dataframe_3d_ago[['日期', '指数代码', '两融资金净买入']],
            g.jingni_csi_lrzj_factor_dataframe_4d_ago[['日期', '指数代码', '两融资金净买入']],
        ], ignore_index=True)
        # 填充空值为0
        index_trading_signal_data.fillna(0, inplace=True)
    print('因子数据：', index_trading_signal_data)

    # 检查因子数据是否为空
    if (g.jingni_csi_qsdf_factor_dataframe_today is None or g.jingni_csi_qsqd_factor_dataframe_today is None or g.jingni_csi_lrzj_factor_dataframe_today is None):
        index_multi_factor_data = None
    else:
        # 合并因子数据 - 分步合并三个DataFrame
        index_multi_factor_data = pandas.merge(g.jingni_csi_qsdf_factor_dataframe_today, g.jingni_csi_qsqd_factor_dataframe_today, on=['日期', '指数代码'], how='outer')
        index_multi_factor_data = pandas.merge(index_multi_factor_data, g.jingni_csi_lrzj_factor_dataframe_today, on=['日期', '指数代码'], how='outer')
        # 填充空值为0
        index_multi_factor_data.fillna(0, inplace=True)
    print('原始因子数据：', index_multi_factor_data)

    g.jingni_csi_multi_factor_selection_dataframe_today = jingni_index_multi_factor_selection({'趋势得分': 0.50, '趋势强度': 0.30, '两融资金净买入': 0.20}, index_multi_factor_data)
    print(g.jingni_csi_multi_factor_selection_dataframe_today)

# 盘中计算申万行业指数多因子分数
def run_jingni_swi_multi_factor_selection(context):

    # 检查因子数据是否为空
    if (g.jingni_swi_qsdf_factor_dataframe_today is None or g.jingni_swi_qsqd_factor_dataframe_today is None or g.jingni_swi_lrzj_factor_dataframe_today is None):
        index_multi_factor_data = None
    else:
        # 合并因子数据 - 分步合并三个DataFrame
        index_multi_factor_data = pandas.merge(g.jingni_swi_qsdf_factor_dataframe_today, g.jingni_swi_qsqd_factor_dataframe_today, on=['日期', '指数代码'], how='outer')
        index_multi_factor_data = pandas.merge(index_multi_factor_data, g.jingni_swi_lrzj_factor_dataframe_today, on=['日期', '指数代码'], how='outer')
        # 填充空值为0
        index_multi_factor_data.fillna(0, inplace=True)
    print('原始因子数据：', index_multi_factor_data)
    
    g.jingni_swi_multi_factor_selection_dataframe_today = jingni_index_multi_factor_selection({'趋势得分': 0.50, '趋势强度': 0.30, '两融资金净买入': 0.20}, index_multi_factor_data)
    print(g.jingni_swi_multi_factor_selection_dataframe_today)

# 盘中计算中证行业指数择时信号
def run_jingni_csi_trading_signal_selection(context):

    # 检查因子数据是否为空
    if (g.jingni_csi_qsdf_factor_dataframe_today is None or g.jingni_csi_qsdf_factor_dataframe_1d_ago is None or g.jingni_csi_qsdf_factor_dataframe_2d_ago is None or g.jingni_csi_qsdf_factor_dataframe_3d_ago is None or g.jingni_csi_qsdf_factor_dataframe_4d_ago is None):
        index_trading_signal_data = None
    else:
        # 纵向合并所有DataFrame
        index_trading_signal_data = pandas.concat([
            g.jingni_csi_qsdf_factor_dataframe_today[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_1d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_2d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_3d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_csi_qsdf_factor_dataframe_4d_ago[['日期', '指数代码', '趋势得分']]
        ], ignore_index=True)
        # 填充空值为0
        index_trading_signal_data.fillna(0, inplace=True)
    print('因子数据：', index_trading_signal_data)
    
    g.jingni_csi_trading_signal_selection_dataframe = jingni_index_trading_signal_selection(index_trading_signal_data)
    print(g.jingni_csi_trading_signal_selection_dataframe)

# 盘中计算申万行业指数择时信号
def run_jingni_swi_trading_signal_selection(context):

    # 检查因子数据是否为空
    if (g.jingni_swi_qsdf_factor_dataframe_today is None or g.jingni_swi_qsdf_factor_dataframe_1d_ago is None or g.jingni_swi_qsdf_factor_dataframe_2d_ago is None or g.jingni_swi_qsdf_factor_dataframe_3d_ago is None or g.jingni_swi_qsdf_factor_dataframe_4d_ago is None):
        index_trading_signal_data = None
    else:
        # 纵向合并所有DataFrame
        index_trading_signal_data = pandas.concat([
            g.jingni_swi_qsdf_factor_dataframe_today[['日期', '指数代码', '趋势得分']],
            g.jingni_swi_qsdf_factor_dataframe_1d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_swi_qsdf_factor_dataframe_2d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_swi_qsdf_factor_dataframe_3d_ago[['日期', '指数代码', '趋势得分']],
            g.jingni_swi_qsdf_factor_dataframe_4d_ago[['日期', '指数代码', '趋势得分']]
        ], ignore_index=True)
        # 填充空值为0
        index_trading_signal_data.fillna(0, inplace=True)
    print('因子数据：', index_trading_signal_data)
    
    g.jingni_swi_trading_signal_selection_dataframe = jingni_index_trading_signal_selection(index_trading_signal_data)
    print(g.jingni_swi_trading_signal_selection_dataframe)

# 盘前执行，准备此策略长期固定的变量
def initialize(context):

    g.__strategy_name = '量化多头行业轮动策略'

    # 设定策略基准收益跟踪对象
    set_benchmark('000906.SS')

    # 定义策略开始执行时间为09:30:00
    g.trade_start_time = datetime.time(9,30,0)

    # 定义策略开盘买入执行时间为09:31:00
    g.trade_buy_time_open = datetime.time(9,31,0)

    # 定义策略开盘回购执行时间为09:50:00
    g.trade_buyback_time_open = datetime.time(9,50,0)

    # 定义策略收盘回购执行时间为14:53:00
    g.trade_buyback_time_close = datetime.time(14,53,0)

    # 定义策略收盘买入执行时间为14:54:00
    g.trade_buy_time_close = datetime.time(14,54,0)

     # 如果业务场景是交易
    if is_trade():

        # 定义策略结束执行时间为14:55:00
        g.trade_end_time = datetime.time(14,55,0)

        # 设置函数每天09:05执行1次
        run_daily(context, strategy_start_send_qywx, time='09:05')

        # 设定函数每天10:00执行1次
        run_daily(context, ipo_security, time='10:00')

        # 设定函数每天14:55执行1次
        run_daily(context, strategy_portfolio_send_qywx, time='14:55')

        # 设定函数每天14:56执行1次
        run_daily(context, treasury_reverse_repurchase, time='14:56')

        # 设定函数每3秒执行1次
        run_interval(context, portfolio, seconds=3)

        # 设定函数每3秒执行1次
        run_interval(context, position, seconds=3)

        # 设定函数每6秒执行1次
        run_interval(context, run_trading_strategy, seconds=6)

    # 如果业务场景是回测
    else:
        # 定义策略结束执行时间为23:59:59
        g.trade_end_time = datetime.time(23,59,59)

        # 设定函数每天15:00执行1次
        run_daily(context, run_jingni_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_csi_qsdf_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_swi_qsdf_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_csi_qsqd_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_swi_qsqd_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_csi_lrzj_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_swi_lrzj_factor, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_csi_multi_factor_selection, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_swi_multi_factor_selection, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_csi_trading_signal_selection, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_jingni_swi_trading_signal_selection, time='15:00')

        # 设定函数每天15:00执行1次
        # run_daily(context, run_trading_strategy, time='15:00')

# 盘前执行，准备此策略每天更新的变量
def before_trading_start(context, data):

    # 初始化策略账户信息函数
    portfolio(context)

    # 初始化组合峰值用于回撤跟踪
    if not hasattr(g, 'peak_value') or g.peak_value < g.assets_value:
        g.peak_value = g.assets_value
    g.stop_loss_triggered = False

    # 日内回购记录列表默认值为空
    g.__portfolio = StockPortfolio()

    # 当日买入标识默认值为未买入
    g.order_buy_flag = False
    # 当日回购标识默认值为未买入
    g.order_buyback_flag = False
    # 当日卖出标识默认值为未卖出
    g.order_sell_flag = False

    # 当日指数代码列表
    g.jingni_swi_index_code_list = get_index_list("SELECT 指数代码 FROM jingni_index_list WHERE 指数序列 > 300 AND 指数序列 < 399 AND 成分日期 <= %s ORDER BY 指数序列" % ptrade_trade_cal(-1))
    g.jingni_csi_index_code_list = get_index_list("SELECT 指数代码 FROM jingni_index_list WHERE 指数序列 > 400 AND 指数序列 < 499 AND 成分日期 <= %s ORDER BY 指数序列" % ptrade_trade_cal(-1))
    g.jingni_index_code_list = g.jingni_swi_index_code_list + g.jingni_csi_index_code_list

    # 当日指数成分股列表
    g.jingni_index_security_code_list = pandas.DataFrame(columns=['指数代码', '股票代码'])
    for jingni_index_code in g.jingni_index_code_list:
        # 转换证券代码前缀字符为小写（指数成分股数据库表名为指数前缀小写）
        # jingni_index_code = security_code_letter_lower(jingni_index_code)
        # 获取证券代码.之前的字符串,即A股六位数字或字母字符
        index_code_char_number = security_code_letter_lower(jingni_index_code)[0:security_code_letter_lower(jingni_index_code).rfind('.')]
        if jingni_index_code[0] == '8':
            # 获取指数成分列表
            stock_code_list = get_stock_list("SELECT p03473_f002 FROM jingni.{0} WHERE `000300` = 1 OR `000905` = 1 OR `000852` = 1 OR p03473_f002 NOT LIKE '%.BJ'".format(index_code_char_number))
        else:
            # 获取指数成分列表
            stock_code_list = get_stock_list("SELECT p03473_f002 FROM jingni.{0} WHERE p03473_f002 NOT LIKE '%.BJ'".format(index_code_char_number))
        # 将当前指数的成分股列表添加到汇总DataFrame中
        for stock_code in stock_code_list:
            # 创建新的行数据
            new_row = pandas.DataFrame({'指数代码': [jingni_index_code], '股票代码': [stock_code]})
            # 追加到汇总DataFrame
            g.jingni_index_security_code_list = pandas.concat([g.jingni_index_security_code_list, new_row], ignore_index=True)

    # 创建全局因子数据
    g.jingni_factor_dataframe = pandas.DataFrame(columns=['日期', '指数代码', '趋势得分', '趋势强度', '两融资金净买入', '多因子分数', '择时信号值'])

    # 计算前1日趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_index_code_list, -1)
    # # 计算前2日趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_index_code_list, -2)
    # # 计算前3日趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_index_code_list, -3)
    # # 计算前4日趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_index_code_list, -4)

    # # 计算前1日申万指数趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_swi_index_code_list, -1)
    # # 计算前2日申万指数趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_swi_index_code_list, -2)
    # # 计算前3日申万指数趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_swi_index_code_list, -3)
    # # 计算前4日申万指数趋势得分因子数据
    # jingni_qsdf_factor(g.jingni_swi_index_code_list, -4)

    # 计算前1日趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_index_code_list, -1)
    # # 计算前2日趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_index_code_list, -2)
    # # 计算前3日趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_index_code_list, -3)
    # # 计算前4日趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_index_code_list, -4)

    # # 计算前1日申万指数趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_swi_index_code_list, -1)
    # # 计算前2日申万指数趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_swi_index_code_list, -2)
    # # 计算前3日申万指数趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_swi_index_code_list, -3)
    # # 计算前4日申万指数趋势强度因子数据
    # jingni_qsqd_factor(g.jingni_swi_index_code_list, -4)

    # 计算前1日两融资金因子数据
    # jingni_lrzj_factor(g.jingni_index_code_list, -1)
    # # 计算前2日两融资金因子数据
    # jingni_lrzj_factor(g.jingni_index_code_list, -2)
    # # 计算前3日两融资金因子数据
    # jingni_lrzj_factor(g.jingni_index_code_list, -3)
    # # 计算前4日两融资金因子数据
    # jingni_lrzj_factor(g.jingni_index_code_list, -4)

    # # 计算前1日申万指数两融资金因子数据
    # jingni_lrzj_factor(g.jingni_swi_index_code_list, -1)
    # # 计算前2日申万指数两融资金因子数据
    # jingni_lrzj_factor(g.jingni_swi_index_code_list, -2)
    # # 计算前3日申万指数两融资金因子数据
    # jingni_lrzj_factor(g.jingni_swi_index_code_list, -3)
    # # 计算前4日申万指数两融资金因子数据
    # jingni_lrzj_factor(g.jingni_swi_index_code_list, -4)

    # # 当日中证指数多因子分数数据
    # g.jingni_csi_multi_factor_selection_dataframe_today = None
    # # 当日申万指数多因子分数数据
    # g.jingni_swi_multi_factor_selection_dataframe_today = None

    # # 当日中证指数择时信号值数据
    # g.jingni_csi_trading_signal_selection_dataframe_today = None
    # # 当日申万指数择时信号值数据
    # g.jingni_swi_trading_signal_selection_dataframe_today = None

# 盘中执行，按照此策略的周期参数运行
def handle_data(context, data):
    # trading_strategy(context)
    pass