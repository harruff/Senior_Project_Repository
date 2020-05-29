/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include "../Gym.h"
#include "../../AbstractManager.h"
#include "../RLSharedMemory.h"
#include "../message/gridWorld.pb.h"

namespace MyBot {
	const int BIG_TILEPOSITION_SCALE = 96;

	typedef BWAPI::Point<int, BIG_TILEPOSITION_SCALE> BigTilePosition;

	class RLGridWorld : public BWML::Gym
	{
	private:
		//���Ǽҵ� ����
		int currentEpisode = 0;
		int maxEpisode = 1000;

		static const int MAX_X = 5;
		static const int MAX_Y = 5;
		const Position GOAL = Position(2, 2);
		const Position TRAP[2] = { Position(1, 2), Position(2, 1) };
		vector<vector<float>> q_table;

		int action;
		// (0, 0) �� ���� Position (�߾�)
		Position startPos;
		// (0, 0) �� ���� BigTilePosition
		BigTilePosition leftTop;
		// agent �� ���� ��ǥ
		Position agentPos = Positions::Origin;
		Unit agent = nullptr;

		// ��ȿ ��ǥ �̳��� ����
		void makeValid(Position &pos);
		// ��ǥ -> ���� Position ���� ����
		Position index2Position(Position pos);
		// ��ǥ -> ���� ���� ����
		static int index2order(Position pos) {
			return pos.x + pos.y * MAX_X;
		}
		// ���� -> ��ǥ �� ����
		static Position order2index(int order) {
			return Position(order % MAX_X, order / MAX_X);
		}

	protected:
		// Gym Override
		void init(::google::protobuf::Message *message) override;
		bool isDone() override;
		void reset(bool isFirstResetCall) override;
		float getReward() override;
		void getObservation(::google::protobuf::Message *stateMsg) override;
		bool isResetFinished() override;
		bool isActionFinished() override;
		void makeInitMessage(::google::protobuf::Message *message) override;
		void makeResetResultMessage(::google::protobuf::Message *message) override;
		void makeStepResultMessage(::google::protobuf::Message *message) override;
		void setRenderData(::google::protobuf::Message *stateMsg) override;
		bool initializeAndValidate() override;

	public:
		RLGridWorld(string shmName, ConnMethod method = SHARED_MEMORY);
		~RLGridWorld() {}

		static RLGridWorld &RLGridWorld::Instance(string shmName = "");

		void step(::google::protobuf::Message *stepReqMsg) override;
		void render() override;
	};
}