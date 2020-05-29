/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "UnitData/UnitData.h"
#include "InformationManager.h"
#include "AbstractManager.h"

namespace MyBot
{
	/// �� ���α׷� ������ ���Ǽ� ����� ���� ���� ȭ�鿡 �߰� �������� ǥ���ϴ� class<br>
	/// ���� Manager ��κ��� ������ ��ȸ�Ͽ� Screen Ȥ�� Map �� ������ ǥ���մϴ�
	class UXManager : public AbstractManager
	{
	private:
		UXManager();

		const int dotRadius = 2;

		// ���� ���� ������ Screen �� ǥ���մϴ�
		void drawGameInformationOnScreen(int x, int y);

		/// APM (Action Per Minute) ���ڸ� Screen �� ǥ���մϴ�
		void drawAPM(int x, int y);

		/// Players ������ Screen �� ǥ���մϴ�
		void drawPlayers();

		/// Player ���� �� (Force) ���� ������ Screen �� ǥ���մϴ�
		void drawForces();

		/// Build ���� ���¸� Screen �� ǥ���մϴ�
		void drawBuildStatusOnScreen(int x, int y);


		/// UnitType �� ��� ������ Screen �� ǥ���մϴ�
		//		void drawUnitStatisticsOnScreen(int x, int y);
		//		UnitData�� �߰��� ���� �ʿ�. gangoku 02.05

		/// Unit �� Id �� Map �� ǥ���մϴ�
		void drawUnitIdOnMap();

		/// Unit �� Target ���� �մ� ���� Map �� ǥ���մϴ�
		void drawUnitTargetOnMap();

		/// Bullet �� Map �� ǥ���մϴ�
		/// Cloaking Unit �� Bullet ǥ�ÿ� ���Դϴ�
		void drawBulletsOnMap();


		/// UnitData ������ Screen �� ǥ���մϴ�
		void drawAllUnitData(int x, int y);


		void drawCoolDown(std::pair<const BWAPI::Unit, MyBot::UnitInfo *> &u);

	protected:
		void updateManager() override;

	public:
		/// static singleton ��ü�� �����մϴ�
		static UXManager 	&Instance();

		/// ��Ⱑ ���۵� �� ��ȸ������ �߰� ������ ����մϴ�
		void onStart();
		void drawUnitHP(UnitInfo *u);
	};

	/// ���� �����Ȳ�� ���� ���� ������� �����Ͽ� ǥ���ϱ� ���� Comparator class
	class CompareWhenStarted
	{
	public:

		CompareWhenStarted() {}

		/// ���� �����Ȳ�� ���� ���� ������� �����Ͽ� ǥ���ϱ� ���� sorting operator
		bool operator() (Unit u1, Unit u2)
		{
			int startedU1 = Broodwar->getFrameCount() - (u1->getType().buildTime() - u1->getRemainingBuildTime());
			int startedU2 = Broodwar->getFrameCount() - (u2->getType().buildTime() - u2->getRemainingBuildTime());
			return startedU1 > startedU2;
		}
	};

}
