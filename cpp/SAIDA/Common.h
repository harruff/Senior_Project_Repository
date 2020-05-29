/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <math.h>
#include <cstdio>
#include <cstdlib>

#include <stdarg.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <array>
#include <ctime>
#include <iomanip>

#include <winsock2.h>
#include <windows.h>

#include <BWAPI.h>

#include "Config.h"
#include "CommandUtil.h"
#include "BWEM/src/bwem.h"      // update the path if necessary

using namespace BWAPI;
using namespace BWAPI::UnitTypes;
using namespace std;
using namespace BWEM;
using namespace BWEM::BWAPI_ext;
using namespace BWEM::utils;

typedef unsigned int word;

namespace {
	auto &theMap = BWEM::BWEMMap::Instance();
	auto &bw = Broodwar;
}

#define S bw->self()
#define E bw->enemy()

#define TIME bw->getFrameCount()

#define GYM 1

namespace MyBot
{
	/// �α� ��ƿ
	namespace Logger
	{
		void appendTextToFile(const string &logFile, const string &msg);
		void appendTextToFile(const string &logFile, const char *fmt, ...);
		void overwriteToFile(const string &logFile, const string &msg);
		void debugFrameStr(const char *fmt, ...);
		void debug(const char *fmt, ...);
		void info(const string fileName, const bool printTime, const char *fmt, ...);
		void error(const char *fmt, ...);
	};

	class SAIDA_Exception
	{
	private:
		unsigned int nSE;
		PEXCEPTION_POINTERS     m_pException;
	public:
		SAIDA_Exception(unsigned int errCode, PEXCEPTION_POINTERS pException) : nSE(errCode), m_pException(pException) {}
		unsigned int getSeNumber() {
			return nSE;
		}
		PEXCEPTION_POINTERS getExceptionPointers() {
			return m_pException;
		}
	};

	/// ���� ��ƿ
	namespace FileUtil {
		/// ���丮 ����
		void MakeDirectory(const char *full_path);
		/// ���� ���� üũ (createYn �� true �̸� ������ ���� ��� ���� ����)
		bool isFileExist(const char *filename, bool createYn = false);
		/// ���� ��ƿ - �ؽ�Ʈ ������ �о���δ�
		string readFile(const string &filename);

		/// ���� ��ƿ - ��� ����� �ؽ�Ʈ ���Ϸκ��� �о���δ�
		void readResults();

		/// ���� ��ƿ - ��� ����� �ؽ�Ʈ ���Ͽ� �����Ѵ�
		void writeResults();

		/// fileName ������ startString ���κ��� endString ���α��� �����Ѵ�.
		void eraseHeader(char *fileName, char *startString, char *endString);
		void addHeader(char *fileName, vector<string> contents);
	}

	namespace CommonUtil {
		string getYYYYMMDDHHMMSSOfNow();
		void pause(int milli);
		void create_process(char *cmd, char *param = NULL, char *currentDirectory = NULL, bool isRelativePath = false);
		DWORD killProcessByName(char *processName);
		DWORD findProcessId(char *processName);
	}
}