##设置R语言的报错为英文
Sys.setenv(LANGUAGE = "en")
##禁止转化为因子
options(stringsAsFactors = FALSE)
##清空环境
rm(list=ls())

getwd()#查看当前工作路径
setwd('E:/zx/R_datas/followup/250323_DUKE_recu_followup_fixed_ensemble_pred')#设置工作路径为，注意R语言不识别反斜杠


######################使用csv生存数据绘制生存曲线OS#######################
#################使用中位数分组High Low###############
#包没安装的可以提前安装一下
library(data.table) # 数据读取使用
library(dplyr) #数据处理使用
library(survival) #生存分析使用
library(survminer) #绘制生存曲线

data1 <- fread("DUKE_final_pred_survival.csv")
data1 <- data1 %>% dplyr::filter(OS.time > 90) #删除随访小于90天的
# 数据筛选
single_survival <- data1 %>% 
  dplyr::select(ID,OS,OS.time, "final_prediction") %>% ## 选择这四列数据
  #dplyr::mutate(group = ifelse(data1$pred > median(data1$pred),"High","Low")) %>% ## 根据中位数进行分组
  mutate(OS.time=round(OS.time/30,2)) %>% ## 将时间以月份形式展示
  na.omit() %>% arrange(final_prediction) ##按照分组进行排序
single_survival$final_prediction <- factor(single_survival$final_prediction,levels = c("1","0"))

## 绘制5年生存曲线
data1 <- data1 %>% dplyr::mutate(OS = ifelse(OS.time > 1826, 0, OS)) %>% #10年3652 5年1826
  dplyr::mutate(OS.time = ifelse(OS.time > 1826, 1826, OS.time))

single_survival <- data1 %>% 
  dplyr::select(ID,OS,OS.time, "final_prediction") %>%
  #dplyr::mutate(group = ifelse(data1[, "label"] > cut_off,"High","Low")) %>%
  mutate(OS.time=round(OS.time/30,2)) %>%
  na.omit() %>% arrange(final_prediction)
single_survival$final_prediction <- factor(single_survival$final_prediction,levels = c("0","1"))

sfit <- survfit(Surv(OS.time, OS) ~ final_prediction, data = single_survival)
p <- ggsurvplot(sfit,
           #pval = TRUE,
           #pval.coord=c(0,72),#pvalue坐标位置
           conf.int = F,#置信区间
           fun = "pct",
           xlab = "Time (Months)",
           palette = "lancet",
             #调色板 默认hue， "grey","npg","aaas","lancet","jco", 
             #"ucscgb","uchicago","simpsons"和"rickandmorty"可选
             #c("red", "black"),
           legend.title = ggplot2::element_blank(),
           legend.labs = c("Negative","Positive"),
           break.time.by = 10,
           #ylim = c(70,100),
           risk.table = T,#下面的表格
           tables.height = 0.2,
           ggtheme = theme_bw())
p


### cox回归分析计算HR
b <- select(data1,c("final_prediction","OS","OS.time"))
data.survdiff <- survdiff(Surv(OS.time, OS) ~ final_prediction,data = b)
p.val = 1 - pchisq(data.survdiff$chisq, length(data.survdiff$n) - 1)
HR = (data.survdiff$obs[2]/data.survdiff$exp[2])/(data.survdiff$obs[1]/data.survdiff$exp[1])
up95 = exp(log(HR) + qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
low95 = exp(log(HR) - qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
ci <- paste0(sprintf("%.3f",HR)," [",sprintf("%.3f",low95),", ",sprintf("%.3f",up95),"]")

p$plot <- p$plot + 
  annotate("text", x = 15, y = 15, 
           label = paste0("P value < 0.0001","\n HR (95% CI) = ",ci),   
           ###添加P和HR 95%CI。 "P value = ",sprintf("%.3f",p.val),
           size = 4.5, color = "black",
           hjust = 0.5,#水平对齐的位置，0左对齐，0.5居中，1右对齐
           vjust = 1#垂直对齐的位置，0左对齐，0.5居中，1右对齐
           )+ 
  theme(text = element_text(size = 15))
p$plot
p



#################绘制RFS曲线###############
#需要以10年为时间点，随访数据超过10年的都是生存的，这时候可以把他们的生存时间改为10年，生存状态改为“存活”。
remove(list = ls())
library(survival)
library(survminer)
data1 <- fread("DUKE_final_pred_survival.csv")
data1 <- data1 %>% dplyr::filter(RFI.time > 90) #删除随访小于90天的
## 绘制5年RFS曲线
data1 <- data1 %>% dplyr::mutate(RFI = ifelse(RFI.time > 1826, 0, RFI)) %>% #10年3652 5年1826
  dplyr::mutate(RFI.time = ifelse(RFI.time > 1826, 1826, RFI.time))

single_survival <- data1 %>% 
  dplyr::select(ID,RFI,RFI.time, "final_prediction") %>%
  #dplyr::mutate(group = ifelse(data1[, "label"] > cut_off,"High","Low")) %>%
  mutate(RFI.time=round(RFI.time/30,2)) %>%
  na.omit() %>% arrange(final_prediction)
single_survival$final_prediction <- factor(single_survival$final_prediction,levels = c("0","1"))

sfit <- survfit(Surv(RFI.time, RFI) ~ final_prediction, data = single_survival)
p <- ggsurvplot(sfit,
                #pval = TRUE,
                #pval.coord=c(0,72),#pvalue坐标位置
                conf.int = F,#置信区间
                fun = "pct",
                xlab = "Time (Months)",
                palette = "lancet",
                #调色板 默认hue， "grey","npg","aaas","lancet","jco", 
                #"ucscgb","uchicago","simpsons"和"rickandmorty"可选
                #c("red", "black"),
                legend.title = ggplot2::element_blank(),
                legend.labs = c("Negative","Positive"),
                break.time.by = 10,
                #ylim = c(70,100),
                risk.table = T,#下面的表格
                tables.height = 0.2,
                ggtheme = theme_bw())
p


### cox回归分析计算HR
b <- select(data1,c("final_prediction","RFI","RFI.time"))
data.survdiff <- survdiff(Surv(RFI.time, RFI) ~ final_prediction,data = b)
p.val = 1 - pchisq(data.survdiff$chisq, length(data.survdiff$n) - 1)
HR = (data.survdiff$obs[2]/data.survdiff$exp[2])/(data.survdiff$obs[1]/data.survdiff$exp[1])
up95 = exp(log(HR) + qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
low95 = exp(log(HR) - qnorm(0.975)*sqrt(1/data.survdiff$exp[2]+1/data.survdiff$exp[1]))
ci <- paste0(sprintf("%.3f",HR)," [",sprintf("%.3f",low95),", ",sprintf("%.3f",up95),"]")

p$plot <- p$plot + 
  annotate("text", x = 15, y = 15, 
           label = paste0("P value < 0.0001","\n HR (95% CI) = ",ci),   ###添加P和HR 95%CI
           size = 4.5, color = "black",
           hjust = 0.5,#水平对齐的位置，0左对齐，0.5居中，1右对齐
           vjust = 1#垂直对齐的位置，0左对齐，0.5居中，1右对齐
  )+ 
  theme(text = element_text(size = 15))
p$plot
p


