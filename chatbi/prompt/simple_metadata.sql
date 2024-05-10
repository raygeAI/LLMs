-- 酒店交易信息主表
CREATE TABLE hotel_transaction (
  hotelid BIGINT ,  -- 酒店id
  bizday VARCHAR , -- 酒店营业日
  datekey BIGINT , -- 8位数字日期 20130112 yyyymmdd
  etl_dt VARCHAR , -- 处理批次日期
  dealtime VARCHAR , -- 数据处理时间
  week_cd BIGINT ,     -- 第几周
  month_cd BIGINT ,  -- 月份
  year_cd BIGINT ,     -- 年份
  hoteltypename VARCHAR , -- 管理类型
  hasbudget VARCHAR ,   -- 预算店标签
  single_label_cd DOUBLE , -- 征用店标签
  special_label_name DOUBLE , -- 特殊标签
  hotel_manage_status_cd VARCHAR , -- 经营生命周期状态
  brand_cd VARCHAR , -- 酒店的品牌
  home_id BIGINT , -- 大盘id
  region_id BIGINT , -- 小区id
  large_area_id BIGINT , -- 区域id，和大区id 一样
  branch_region_id BIGINT , -- 大区id
  theater_region_id BIGINT , -- 事业部id
  room_night_unconvert DOUBLE ,  -- 综合出租数，不折算
  roomrevenue DOUBLE , -- 客房收入
  room_night DOUBLE , -- 经过折算后的综合出租数(间夜数)
  hr_room_night DOUBLE , -- 钟点房出租数
  overnight_room_night DOUBLE , -- 过夜出租数(过夜间夜数)
  total_revenue DOUBLE , -- 总收入
  other_revenue DOUBLE , -- 其他收入
  memcard_revenue DOUBLE , -- 会员卡收入
  hr_room_revenue DOUBLE , -- 钟点房收入
  overnight_room_revenue DOUBLE , -- 过夜客房收入
  total_card_cnt BIGINT , -- 总售卡数量
  silve_card_cnt BIGINT , -- 银卡售卡数量
  gold_card_cnt BIGINT , -- 金卡售卡数量
  black_gold_card_cnt BIGINT , -- 黑金卡售卡数量
  individual_guest_room_night DOUBLE , -- 散客出租数(折算)
  member_guest_room_night DOUBLE , -- 会员出租数(折算)
  agr_guest_room_night DOUBLE , -- 协议客出租数(折算)
  agent_guest_room_night DOUBLE , -- 中介出租数(折算)
  group_guest_room_night DOUBLE , -- 团体组出租数(折算)
  longterm_guest_room_night DOUBLE , -- 长包出租数(折算)
  direct_sales_guest_room_night DOUBLE , -- 直销出租数(折算)
  individual_guest_room_night_unconvert BIGINT , -- 散客出租数(不折算)
  member_guest_room_night_unconvert BIGINT , -- 会员出租数(不折算)
  agr_guest_room_night_unconvert BIGINT , -- 协议客出租数(不折算)
  agent_guest_room_night_unconvert BIGINT , -- 中介出租数(不折算)
  group_guest_room_night_unconvert BIGINT , -- 团体组出租数(不折算)
  longterm_guest_room_night_unconvert BIGINT , -- 长包出租数(不折算)
  direct_sales_guest_room_night_unconvert BIGINT , -- 直销出租数(不折算)
  official_book_room_night DOUBLE , --  官渠预定出租数(折算)
  official_book_room_night_unconvert BIGINT , -- 官渠预定出租数(不折算)
  central_book_room_night DOUBLE , -- 中央渠道出租数(折算)
  central_book_room_night_unconvert BIGINT , -- 中央渠道出租数(不折算)
  mobile_book_room_night DOUBLE , -- 移动端出租数(折算)
  mobile_book_room_night_unconvert BIGINT , -- 移动端出租数(不折算)
  app_book_room_night DOUBLE , -- app 出租数(折算)
  wechat_book_room_night DOUBLE , -- 微信出租数(折算)
  minipro_book_room_night DOUBLE , -- 小程序出租数(折算)
  ota_dirtct_book_room_night DOUBLE , -- OTA直连出租数(折算)
  offcial_web_book_room_night DOUBLE , -- 官网渠道出租数(折算)
  offline_store_book_room_night DOUBLE , -- 线下门店出租数(折算)
  book_400_room_night DOUBLE , -- 400 出租数(折算)
  other_book_room_night DOUBLE , -- 其他渠道出租数(折算)
  app_book_room_night_unconvert BIGINT , -- app 出租数(不折算)
  wechat_book_room_night_unconvert BIGINT , -- 微信渠道出租数(不折算)
  minipro_book_room_night_unconvert BIGINT , -- 微信小程序出租数(不折算)
  ota_dirtct_book_room_night_unconvert BIGINT , -- OTA直连出租数(不折算)
  offcial_web_book_room_night_unconvert BIGINT , -- 官网渠道出租数(不折算)
  offline_store_book_room_night_unconvert BIGINT , -- 线下门店出租数(不折算)
  book_400_room_night_unconvert BIGINT , -- 400 出租数(不折算)
  other_book_room_night_unconvert BIGINT , -- 其他渠道出租数(不折算)
  dayscurrmonth BIGINT , -- 当月天数
  nf07_today_receipts DOUBLE , -- 805 总收入
  breakfastfree_revenue DOUBLE , -- 免早餐收入
  pt BIGINT -- 时间字段分区
);

-- 酒店维度信息表，维度id 和维度名称映射表
CREATE TABLE hotel_dim (
  chain_id BIGINT ,
  hotelid BIGINT , -- 酒店id
  hotel_name VARCHAR , -- 酒店名称
  hotel_type_id BIGINT ,
  hotel_type_name VARCHAR ,
  open_date VARCHAR , -- 酒店开业日期
  open_year BIGINT , -- 酒店开业年份
  open_month BIGINT , -- 酒店开业月份
  hotel_age BIGINT , -- 开业天数
  hotel_age_range_id BIGINT ,
  is_new_hotel VARCHAR ,
  hotel_status_id BIGINT ,
  hotel_status_name VARCHAR ,
  brand_cd VARCHAR , -- 酒店品牌代码
  brand_name VARCHAR , -- 酒店品牌名称
  open_no DOUBLE ,
  country_id BIGINT ,
  province_id BIGINT ,
  city_id BIGINT ,
  city_area_id BIGINT ,
  has_budget VARCHAR ,
  first_bizday VARCHAR ,
  is_open_presale VARCHAR ,
  presale_open_dt VARCHAR ,
  scene_id BIGINT ,
  isnewsystem BIGINT ,
  is_current BIGINT ,
  province_name VARCHAR ,
  city_name VARCHAR ,
  city_area_name VARCHAR ,
  bus_db_name VARCHAR ,
  satisfy_rate DOUBLE ,
  region_id BIGINT ,
  region_name VARCHAR ,
  large_area_id BIGINT ,
  large_area_name VARCHAR ,
  branch_region_id BIGINT ,
  branch_region_name VARCHAR ,
  theater_region_id BIGINT ,
  theater_region_name VARCHAR ,
  hotel_address VARCHAR ,
  hotel_tel VARCHAR ,
  hotellivestatus1_id DOUBLE ,
  hotellivestatus1_name DOUBLE ,
  hotellivestatus2_id DOUBLE ,
  hotellivestatus2_name DOUBLE ,
  hotellivestatus3_id DOUBLE ,
  hotellivestatus3_name DOUBLE ,
  hotellivestatus4_id DOUBLE ,
  hotellivestatus4_name DOUBLE ,
  hotelbillingstatus_id BIGINT ,
  hotelbillingstatus_name VARCHAR ,
  weixin_latitude DOUBLE ,
  weixin_longitude DOUBLE ,
  baidu_longitude DOUBLE ,
  baidu_latitude DOUBLE ,
  hotel_facility VARCHAR ,
  room_facility VARCHAR ,
  hotel_introduction VARCHAR ,
  city_level VARCHAR ,
  manager_appoint_type VARCHAR ,
  manager_appoint_type_desc VARCHAR ,
  hotel_live_statusi_cd VARCHAR ,
  hotel_live_statusi_name VARCHAR ,
  hotel_live_statusii_cd VARCHAR ,
  hotel_live_statusii_name VARCHAR ,
  hotel_manage_status_cd VARCHAR ,
  hotel_manage_status_name VARCHAR ,
  contract_room_count DOUBLE ,
  hotel_manage_status_cd_new VARCHAR ,
  hotel_manage_status_name_new VARCHAR ,
  hotel_manager_id DOUBLE ,
  hotel_manager_name VARCHAR ,
  special_business_label_cd VARCHAR ,
  special_business_label_name DOUBLE ,
  gxt_region_id BIGINT ,
  gxt_region_name VARCHAR ,
  single_label_cd DOUBLE ,
  single_label_name DOUBLE ,
  appoint_type_operate VARCHAR ,
  szt_hotel_age BIGINT ,
  szt_business_name VARCHAR ,
  szt_hotel_mgr_id DOUBLE ,
  szt_hotel_mgr_name VARCHAR ,
  szt_hotel_mgr_entry_date VARCHAR ,
  szt_hotel_mgr_onjob_date VARCHAR ,
  szt_hotel_employee_cnt BIGINT ,
  szt_hotel_employee_able_inter_cnt BIGINT
);

-- hotel_transaction.hotelid can be joined with hotel_dim.hotelid