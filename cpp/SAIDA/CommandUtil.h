/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "Config.h"

namespace MyBot
{
	struct Rect
	{
		int x, y;
		int height, width;
	};

	/// �̵� (move), ���� (attack), ���� (repair), ��Ŭ�� (rightClick)  �� ���� ��Ʈ�� ����� ���� �� ���� üũ�ؾ��� ���׵��� üũ�� �� ��� �������� �ϴ� ���� �Լ���
	namespace CommandUtil
	{
		/// attacker �� target �� �����ϵ��� ��� �մϴ�
		bool attackUnit(BWAPI::Unit attacker, BWAPI::Unit target, bool repeat = false);

		/// attacker �� targetPosition �� ���� ���� ������ ��� �մϴ�
		bool attackMove(BWAPI::Unit attacker, const BWAPI::Position &targetPosition, bool repeat = false);

		/// attacker �� targetPosition �� ���� �̵� ������ ��� �մϴ�
		void move(BWAPI::Unit attacker, const BWAPI::Position &targetPosition, bool repeat = false);

		/// unit �� target �� ���� � ������ �ϵ��� ��� �մϴ�<br>
		/// �ϲ� ������ Mineral Field ���� : Mineral �ڿ� ä��<br>
		/// �ϲ� ������ Refinery �ǹ����� : Gas �ڿ� ä��<br>
		/// ���� ������ �ٸ� �Ʊ� ���ֿ��� : Move ���<br>
		/// ���� ������ �ٸ� ���� ���ֿ��� : Attack ���<br>
		void rightClick(BWAPI::Unit unit, BWAPI::Unit target, bool repeat = false, bool rightClickOnly = false);
		void rightClick(BWAPI::Unit unit, BWAPI::Position target, bool repeat = false);

		/// unit �� target �� ���� ���� �ϵ��� ��� �մϴ�
		void repair(BWAPI::Unit unit, BWAPI::Unit target, bool repeat = false);

		void patrol(BWAPI::Unit patroller, const BWAPI::Position &targetPosition, bool repeat = false);
		void hold(BWAPI::Unit holder, bool repeat = false);
		void holdControll(BWAPI::Unit unit, BWAPI::Unit target, BWAPI::Position targetPosition, bool targetUnit = false);
		bool build(BWAPI::Unit builder, BWAPI::UnitType building, BWAPI::TilePosition buildPosition);
		void gather(BWAPI::Unit worker, BWAPI::Unit target);
	};


	namespace UnitUtil
	{
		bool IsCombatUnit(BWAPI::Unit unit);
		bool IsValidUnit(BWAPI::Unit unit);
		bool CanAttack(BWAPI::Unit attacker, BWAPI::Unit target);
		double CalculateLTD(BWAPI::Unit attacker, BWAPI::Unit target);
		// attacker �� target �� �����Ҷ� ��Ÿ��� ��ȯ�Ѵ�. (���׷��̵� ����, ��Ŀ�� ���� ��Ÿ� ������ �ݿ� �ȵ�.)
		// ���� Unit ���� ����ϴ� ���, �þ߿��� ������� �߸��� �� ��ȯ
		int GetAttackRange(BWAPI::Unit attacker, BWAPI::Unit target);
		int GetAttackRange(BWAPI::Unit attacker, bool isTargetFlying);
		int GetAttackRange(BWAPI::UnitType attackerType, BWAPI::Player attackerPlayer, bool isFlying);

		// attacker �� target �� �����Ҷ� ����ϴ� weaponType �� ��ȯ�Ѵ�.
		// ���� Unit ���� ����ϴ� ���, �þ߿��� ������� �߸��� �� ��ȯ
		BWAPI::WeaponType GetWeapon(BWAPI::Unit attacker, BWAPI::Unit target);
		BWAPI::WeaponType GetWeapon(BWAPI::Unit attacker, bool isTargetFlying);
		BWAPI::WeaponType GetWeapon(BWAPI::UnitType attackerType, bool isFlying);

		size_t GetAllUnitCount(BWAPI::UnitType type);

		BWAPI::Unit GetClosestUnitTypeToTarget(BWAPI::UnitType type, BWAPI::Position target);
		double GetDistanceBetweenTwoRectangles(Rect &rect1, Rect &rect2);

		BWAPI::Position GetAveragePosition(std::vector<BWAPI::Unit>  units);
		BWAPI::Unit GetClosestEnemyTargetingMe(BWAPI::Unit myUnit, std::vector<BWAPI::Unit>  units);

		BWAPI::Position getPatrolPosition(BWAPI::Unit attackUnit, BWAPI::WeaponType weaponType, BWAPI::Position targetPos);
	};
}