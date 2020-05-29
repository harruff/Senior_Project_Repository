/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once
#include <string>
#include "SharedMemory.h"
#include "../AbstractManager.h"

using namespace std;
using namespace BWML;

#define WIN_REWARD 100
#define LOSE_REWARD -100

namespace BWML {
	class Gym : public MyBot::AbstractManager
	{
	private:
		void updateManager() override final;
		bool isInitialized;
		bool isResetting;
		bool isActing;

		void setInitialzing() {
			isInitialized = true;
			isActing = false;
			isResetting = false;
		}

		void setActing() {
			isActing = true;
			isResetting = false;
		}

		void setResetting() {
			isActing = false;
			isResetting = true;
		}

	protected:
		// TODO ���� ��Ÿ���� ���ۼ� ���� ��ü �ʿ�
		SharedMemory *connection;
		::google::protobuf::Message *initReqMsg;
		::google::protobuf::Message *initResMsg;
		::google::protobuf::Message *resetResMsg;
		::google::protobuf::Message *stepReqMsg;
		::google::protobuf::Message *stepResMsg;
		::google::protobuf::Message *renderReqMsg;
		::google::protobuf::Message *renderResMsg;

		int episodeNum = 0;
		int stepNum = 0;
		int startFrame;
		vector<bool> isInvalidAction;

		// step �� ������(default 6)
		int STEP_FRAME = 6;

		virtual void init(::google::protobuf::Message *message) = 0;
		virtual void reset(bool isFirstResetCall) = 0;
		virtual bool isResetFinished() = 0;
		virtual bool isActionFinished() = 0;
		virtual float getReward() = 0;
		virtual bool isDone() = 0;
		// python ���� ������ �ʱⰪ ����
		virtual void makeInitMessage(::google::protobuf::Message *message) = 0;
		// python ���� ������ step �� ����� ����
		virtual void makeResetResultMessage(::google::protobuf::Message *message);
		virtual void makeStepResultMessage(::google::protobuf::Message *message);
		virtual void getInformation(::google::protobuf::Message *infoMsg);
		virtual void getObservation(::google::protobuf::Message *stateMsg) = 0;
		virtual void setRenderData(::google::protobuf::Message *stateMsg) {};

		/// update �� ���� ���� �� �ʱ�ȭ �� ������ ���ռ� üũ�� �� �ش�.
		/// return ���� update �� �����ϴ��� ����
		virtual bool initializeAndValidate() = 0;

		const bool getIsInitialized() {
			return isInitialized;
		}

		const bool getIsResetting() {
			return isResetting;
		}

		const bool getIsActing() {
			return isActing;
		}

	public:
		Gym(string name) : AbstractManager(name) {
			connection = nullptr;
			isInitialized = false;
			isResetting = false;
			isActing = false;
			startFrame = 0;

			Message::InitReq initReq;
			initReqMsg = initReq.New();
			Message::InitRes initRes;
			initResMsg = initRes.New();
			Message::ResetRes resetRes;
			resetResMsg = resetRes.New();
			Message::StepReq stepReq;
			stepReqMsg = stepReq.New();
			Message::StepRes stepRes;
			stepResMsg = stepRes.New();
			Message::RenderReq renderReq;
			renderReqMsg = renderReq.New();
			Message::RenderRes renderRes;
			renderResMsg = renderRes.New();
		};
		virtual ~Gym() {
			if (connection) {
				connection->close();
			}

			deleteObject(connection);
			deleteObject(initReqMsg);
			deleteObject(initResMsg);
			deleteObject(resetResMsg);
			deleteObject(stepReqMsg);
			deleteObject(stepResMsg);
		}
		void initialize();
		void doReset(bool isFirstResetCall = true);
		virtual void step(::google::protobuf::Message *stepReqMsg) = 0;
		virtual void render() = 0;
		void parseInitReq(char *message, int length);
		void parseStepReq(char *message, int length);
		void parseRenderReq(char *message, int length);

		void deleteObject(void *obj) {
			if (obj)
				delete obj;
		}
		void sendMessage(char *operation, ::google::protobuf::Message *message) {
			connection->sendMessage(operation, message);
		}

		bool receiveMessage() {
			return connection->receiveMessage();
		}

		void close() {
			if (connection)
				connection->close();
		}
	};
}