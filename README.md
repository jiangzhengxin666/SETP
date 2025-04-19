# SETP
Spatial enhanced trajectory planner

## 目录

- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [使用方法](#使用方法)

## 快速开始

1. 前往谷歌云盘https://drive.google.com/file/d/16bKJiA3eBdqNl08PC8HDs8M9cC-3Aav5/view?usp=sharing
处下载模型权重。
2. 前往项目的datasets/下载数据集，更改图片的路径位置。

## 安装指南

本项目以来模型库和Qwen2VL-2b相同。本文模型调用方式和Qwen2VL-2b相同。提供调用范例脚本src/qwen2QuickStart.py

## 使用方法

1. 可以使用src/modelEvaluateTrack.py文件对模型的轨迹规划能力进行测试，本模型的测试结果已保存在results/trackNusceneResults.json文件中。
2. 可以使用src/trackDataProcess.py计算模型的轨迹误差。
