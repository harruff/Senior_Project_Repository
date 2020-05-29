/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "BWAPI.h"
#include <cassert>

// minwindef ����
#define MAX_PATH 260

/// �� ���α׷� ����
namespace Config
{
	/// ���� ���� ����
	namespace Files
	{
		/// �α� ���� �̸�
		extern std::string LogFilename;
		/// Ÿ�Ӿƿ� ���� �̸�
		extern std::string TimeoutFilename;
		/// �����α� ���� �̸�
		extern std::string ErrorLogFilename;
		/// �б� ���� ���
		extern std::string ReadDirectory;
		/// ���� ���� ���
		extern std::string WriteDirectory;
		/// SAIDA.exe ������ �ִ� ����
		extern char saidaDirectory[MAX_PATH];
		/// saida.ini path
		extern char saida_ini_filename[MAX_PATH];
		/// Starcraft HOME ����
		extern char StarcraftDirectory[MAX_PATH];
		/// bwapi.ini path
		extern char bwapi_ini_filename[MAX_PATH];

		void initialize();
	}

	/// CommonUtil ���� ����
	namespace Tools
	{
		/// MapGrid ���� �� �� GridCell �� size
		extern int MAP_GRID_SIZE;
	}

	/// BWAPI �ɼ� ���� ����
	namespace BWAPIOptions
	{
		/// ���ÿ��� ������ ������ �� ���ӽ��ǵ� (�ڵ� ���� �� �������� ������ ������ ���� ���� ������ �����)<br>
		/// Speedups for automated play, sets the number of milliseconds bwapi spends in each frame.<br>
		/// Fastest: 42 ms/frame.  1�ʿ� 24 frame. �Ϲ������� 1�ʿ� 24frame�� ���� ���Ӽӵ��� �մϴ�.<br>
		/// Normal: 67 ms/frame. 1�ʿ� 15 frame.<br>
		/// As fast as possible : 0 ms/frame. CPU�� �Ҽ��ִ� ���� ���� �ӵ�.
		extern int SetLocalSpeed;
		/// ���ÿ��� ������ ������ �� FrameSkip (�ڵ� ���� �� �������� ������ ������ ���� ���� ������ �����)<br>
		/// frameskip�� �ø��� ȭ�� ǥ�õ� ������Ʈ ���ϹǷ� �ξ� �����ϴ�
		extern int SetFrameSkip;
		/// rendering on/off
		extern bool EnableGui;
		/// ���ÿ��� ������ ������ �� ����� Ű����/���콺 �Է� ��� ���� (�ڵ� ���� �� �������� ������ ������ ���� ���� ������ �����)
		extern bool EnableUserInput;
		/// ���ÿ��� ������ ������ �� ��ü ������ �� ���̰� �� ������ ���� (�ڵ� ���� �� �������� ������ ������ ���� ���� ������ �����)
		extern bool EnableCompleteMapInformation;
		/// ������ ����� �����ϰ� ����.
		extern bool RestartGame;
		/// ���� ����(leaveGame) �Ŀ� �߰��� ����Ǵ� on frame �ȵ����� ��.
		extern bool EndGame;
	}

	/// ����� ���� ����
	namespace Debug
	{
		/// ȭ�� ǥ�� ���� - ���� ����
		extern bool DrawGameInfo;

		/// ȭ�� ǥ�� ���� - ����
		extern bool DrawBWEMInfo;

		/// ȭ�� ǥ�� ���� - ���� ~ Target �� ����
		extern bool DrawUnitTargetInfo;



		/// ȭ�� ǥ�� ���� - ���� ����
		extern bool DrawScoutInfo;

		/// ȭ�� ǥ�� ���� - ���콺 Ŀ��
		extern bool DrawMouseCursorInfo;

		/// ȭ�� ǥ�� ���� - AllUnitVector, Map Information
		extern bool DrawMyUnit;
		extern bool DrawEnemyUnit;

		extern bool DrawLastCommandInfo;
		extern bool DrawUnitStatus;

		extern bool Focus;
		extern bool Console_Log;
	}

	// �⺻ ���� ����
	namespace Propeties
	{
		/// �ڿ� �����ÿ� ���Ǵ� measure duration �����̸� ������ seconds
		extern int duration;
		extern bool recoring;

	}

}