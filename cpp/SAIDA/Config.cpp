/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "Config.h"
#include "Common.h"

namespace Config
{
	// BasicBot 1.1 Patch Start ////////////////////////////////////////////////
	// �� �̸� �� ���� ��� �⺻�� ����

	namespace Files
	{
		std::string LogFilename;
		std::string TimeoutFilename;
		std::string ErrorLogFilename;
		std::string ReadDirectory = "bwapi-data\\read\\";
		std::string WriteDirectory = "bwapi-data\\write\\";
		char saidaDirectory[MAX_PATH];
		char saida_ini_filename[MAX_PATH];
		char StarcraftDirectory[MAX_PATH] = "c:\\starcraft\\";
		char bwapi_ini_filename[MAX_PATH];
		void initialize() {
			// Config::Files::saidaDirectory : saida.exe �������� ��ġ ����
			GetModuleFileName(NULL, Config::Files::saidaDirectory, MAX_PATH);

			for (int i = strlen(Config::Files::saidaDirectory); i >= 0 && Config::Files::saidaDirectory[i] != '\\'; i--)
				Config::Files::saidaDirectory[i] = '\0';

			// Config::Files::saida_ini_filename : saida.ini ��ġ ����
			strcpy_s(Config::Files::saida_ini_filename, MAX_PATH, Config::Files::saidaDirectory);

			if (!MyBot::FileUtil::isFileExist(((string)Config::Files::saidaDirectory + "SAIDA.ini").c_str()))
				if (MyBot::FileUtil::isFileExist(((string)Config::Files::saidaDirectory + "..\\..\\SAIDA.ini").c_str()))
					strcat_s(Config::Files::saida_ini_filename, MAX_PATH, "..\\..\\");

			strcat_s(Config::Files::saida_ini_filename, MAX_PATH, "SAIDA.ini");

			if (!MyBot::FileUtil::isFileExist(Config::Files::saida_ini_filename, true)) {
				cout << "SAIDA.ini file is not exist." << endl;

				WritePrivateProfileString("default", "STARCRAFT_HOME", Config::Files::StarcraftDirectory, Config::Files::saida_ini_filename);
				WritePrivateProfileString("auto_menu", "map_path", "maps/usemap/", Config::Files::saida_ini_filename);

				cout << "SAIDA.ini file created." << endl;
			}

			// Config::Files::StarcraftDirectory : STARCRAFT_HOME ����
			GetPrivateProfileString("default", "STARCRAFT_HOME", Config::Files::StarcraftDirectory, Config::Files::StarcraftDirectory, MAX_PATH, Config::Files::saida_ini_filename);

			// Config::Files::bwapi_ini_filename : bwapi.ini ��ġ ����
			strcpy_s(Config::Files::bwapi_ini_filename, MAX_PATH, Config::Files::StarcraftDirectory);
			strcat_s(Config::Files::bwapi_ini_filename, MAX_PATH, "bwapi-data\\bwapi.ini");

			// Config::Files::ReadDirectory, WriteDirectory ����
			ReadDirectory = Config::Files::StarcraftDirectory + ReadDirectory;
			WriteDirectory = Config::Files::StarcraftDirectory + WriteDirectory;
		}
	}

	// BasicBot 1.1 Patch End //////////////////////////////////////////////////

	namespace BWAPIOptions
	{
		int SetLocalSpeed = 0;
		int SetFrameSkip = 0;
		bool EnableGui = true;
		bool EnableUserInput = true;
		bool EnableCompleteMapInformation = false;
		bool RestartGame = true;
		bool EndGame = false;
	}

	namespace Tools
	{
		extern int MAP_GRID_SIZE = 32;
	}

	namespace Debug
	{
		bool DrawGameInfo = true;
		bool DrawScoutInfo = true;
		bool DrawMouseCursorInfo = true;
		bool DrawBWEMInfo = true;
		bool DrawUnitTargetInfo = false;
		// �Ʒ��� ���� �Ѱ�����
		bool DrawMyUnit = false;
		bool DrawEnemyUnit = false;
		bool DrawLastCommandInfo = false;
		bool DrawUnitStatus = true;

		bool Focus = true;
		bool Console_Log = false;
	}


	// �⺻ ���� ����
	namespace Propeties
	{
		/// �ڿ� �����ÿ� ���Ǵ� measure duration �����̸� ������ seconds
		int duration = 1;
		bool recoring = true;

	}

}