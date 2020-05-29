/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "Gym.h"
#include "SharedMemory.h"
#include "RLSharedMemory.h"

#include "message/common.pb.h"

#define MAX_GYM_NAME_LENGTH 30 // Gym �̸��� �ִ� �ڸ���

namespace BWML {
	enum class AIType {
		EMBEDED, DLL, EXE, HUMAN
	};

	class GymFactory
	{
	private:
		Gym *gym;
		// TODO ���� namespace �� conn ������ ����. sharedmemory �� zmq ���� �ڽ����� �ϴ� �θ� Ŭ������ ���� �ʿ�.
		SharedMemory *connection;

		// Gym �� �߰��Ǹ� ����Ǿ�� �� �޼ҵ�
		void String2Gym(string gymName, string shmName, int version);
		string mapName;
		string autoMenuMode = "SINGLE_PLAYER";
		string enemyBot = "";
		AIType enemyType = AIType::EMBEDED;

		bool autoKillStarcraft = true;

	public:
		GymFactory() {
			gym = nullptr;
		}
		~GymFactory() {}

		static GymFactory &Instance();

		void Initialize(ConnMethod method = SHARED_MEMORY);

		Gym *GetGym() {
			return gym;
		}

		void InitializeGym() {
			gym->initialize();
		}

		void Destroy();
	};
}

