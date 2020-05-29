/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#include "SharedMemoryManager.h"

using namespace BWML;
using namespace MyBot;

SharedMemoryManager::~SharedMemoryManager() {
	for (auto s : shmList) {
		s->close();
	}

	shmList.clear();
}

SharedMemoryManager &SharedMemoryManager::Instance() {
	static SharedMemoryManager sharedMemoryManager;
	return sharedMemoryManager;
}

bool SharedMemoryManager::CreateMemoryMap(SharedMemory *shm)
{
	cout << "���� �޸� ���� ����..(" << shm->getMapName() << ")" << endl;

	bool isCreateSAIDAIPC = false;

	// �� ���µ� �����޸𸮰� �ִ��� Ȯ��.
	shm->hFileMap = OpenFileMapping(FILE_MAP_ALL_ACCESS, false, shm->getMapName());

	// �����޸� ���� ���ϴ� ��� ����.
	if (!shm->hFileMap) {
		cout << "���� �޸� ���� ����..." << endl;

		shm->hFileMap = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, shm->MAX_SHM_SIZE, shm->getMapName());

		// ���� ���� �� ���� (�޸� ����, ���� ��)
		if (!shm->hFileMap) {
			Logger::error("�����޸� ���� ����!");
			return false;
		}
		else {
			cout << "���� �޸� ���� ���� ����" << endl;
			isCreateSAIDAIPC = true;
		}
	}
	else {
		cout << "���� �޸� �̹� ����" << endl;
	}

	if ((shm->pData = (char *)MapViewOfFile(shm->hFileMap, FILE_MAP_ALL_ACCESS, 0, 0, shm->MAX_SHM_SIZE)) == NULL) {
		Logger::error("�����޸� �ݱ�");
		CloseHandle(shm->hFileMap);
		return false;
	}
	else
	{
		// ���� �������� �ʱ�ȭ
		if (isCreateSAIDAIPC)
			memset(shm->pData, NULL, shm->MAX_SHM_SIZE);
	}

	shmList.push_back(shm);

	return true;
}

void SharedMemoryManager::FreeMemoryMap(SharedMemory *shm) {
	cout << "FreeMemoryMap" << endl;
	auto del = find_if(shmList.begin(), shmList.end(), [shm](SharedMemory * s) {
		return shm->getMapName() == s->getMapName();
	});

	if (del != shmList.end()) {
		if (shm && shm->hFileMap) {
			if (shm->pData)
				UnmapViewOfFile(shm->pData);

			if (shm->hFileMap)
				CloseHandle(shm->hFileMap);
		}

		shmList.erase(del);
	}
}
