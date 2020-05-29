/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "Common.h"
#include "InformationManager.h"

#define NOT_DANGER 1000

namespace MyBot
{
	vector<Position> getWidePositions(Position source, Position target, bool forward = true, int gap = TILE_SIZE, int angle = 30, int cnt = 5);
	vector<Position> getRoundPositions(Position source, int gap = TILE_SIZE, int angle = 30);
	Position getDirectionDistancePosition(Position source, Position direction, int distance = TILE_SIZE);

	// Back Position ���� API
	Position getBackPostion(UnitInfo *unit, Position target, int length, bool avoidUnit = false);
	void moveBackPostion(UnitInfo *unit, Position ePos, int length);
	bool isValidPath(Position st, Position en);
	int getPathValue(Position st, Position en);
	int getPathValueForAir(Position en);
	int getGroundDistance(Position st, Position en);
	int getAltitude(Position pos);
	int getAltitude(TilePosition pos);
	int getAltitude(WalkPosition pos);

	// ���� ���� ��� ���� �� ȸ�� �̵� �ڵ�
	//void GoWithoutDamage(Unit unit, Position pos);
	//void makeLine_dh(Unit unit, Unit target, double *m, Position pos);
	//void drawLine_dh(Position unit, double m);

	bool isUseMapSettings();
	void focus(Position pos);
	void restartGame();
	void leaveGame();

	bool isSameArea(UnitInfo *u1, UnitInfo *u2);
	bool isSameArea(const Area *a1, const Area *a2);
	bool isSameArea(TilePosition a1, TilePosition a2);
	bool isSameArea(Position a1, Position a2);

	// �� ������ ���� �����ִ��� üũ�Ѵ�.
	bool isBlocked(Unit unit, int size = 32);
	bool isBlocked(const UnitType unitType, Position centerPosition, int size = 32);
	bool isBlocked(const UnitType unitType, TilePosition topLeft, int size = 32);
	bool isBlocked(int top, int left, int bottom, int right, int size = 32);

	// UnitList�� ��� Postion ��
	Position getAvgPosition(uList units);
	// �ش� Position�� +1, -1 Tile ������ Random Position
	Position findRandomeSpot(Position p);

	// Unit�� target���� Tower�� ���ذ��� Function : direction�� ��/�Ʒ� ��.
	// ���� ���� ���� ���� ���̸� Positions::None�� return��.
	// direction �� 1�� �ð����, -1�� �ݽð����
	bool goWithoutDamage(Unit u, Position target, int direction, int dangerGap = 3 * TILE_SIZE);
	void kiting(UnitInfo *attacker, UnitInfo *target, int dangerPoint, int threshold);
	void attackFirstkiting(UnitInfo *attacker, UnitInfo *target, int dangerPoint, int threshold);
	void pControl(UnitInfo *attacker, UnitInfo *target);
	UnitInfo *getGroundWeakTargetInRange(UnitInfo *attacker, bool worker = false);

	// �߽ɰ� ����(�� ����)�� ������ �� ���� �־����� ������ �������� ������ŭ ������ ������ �� �ٸ� �� ���� ��ǥ�� ���մϴ�.
	Position getCirclePosFromPosByDegree(Position center, Position fromPos, double degree);
	// �߽ɰ� ����(���� ����)�� ������ �� ���� �־����� ������ �������� ������ŭ ������ ������ �� �ٸ� �� ���� ��ǥ�� ���մϴ�.
	Position getCirclePosFromPosByRadian(Position center, Position fromPos, double radian);
	// ������ �Ÿ��� ����(�� ����)�� �־����� �� ������ �� ������ �Ÿ���ŭ ������ ���� ��ǥ�� ���մϴ�.
	Position getPosByPosDistDegree(Position pos, int dist, double degree);
	// ������ �Ÿ��� ����(���� ����)�� �־����� �� ������ �� ������ �Ÿ���ŭ ������ ���� ��ǥ�� ���մϴ�.
	Position getPosByPosDistRadian(Position pos, int dist, double degree);
	// p1 �������� p2 �� ����(���� ����)�� ��ȯ�Ѵ�.
	double getRadian(Position p1, Position p2);
	double getRadian2(Position p1, Position p2);

	int getDamage(Unit attacker, Unit target);
	int getDamage(UnitType attackerType, UnitType targetType, Player attackerPlayer, Player targetPlayer);
	UnitInfo *getDangerUnitNPoint(Position p, int *point, bool isFlyer);

	// AttackRange ��� �������� �Ÿ��� �����´�. weaponRange �� �� ����
	int getAttackDistance(int aLeft, int aTop, int aRight, int aBottom, int tLeft, int tTop, int tRight, int tBottom);
	int getAttackDistance(Unit attacker, Unit target);
	int getAttackDistance(Unit attacker, UnitType targetType, Position targetPosition);
	int getAttackDistance(UnitType attackerType, Position attackerPosition, Unit target);
	int getAttackDistance(UnitType attackerType, Position attackerPosition, UnitType targetType, Position targetPosition);

	// ���� ����� ��ֹ������� �Ÿ�
	vector<int> getNearObstacle(UnitInfo *uInfo, int directCnt, bool resource = false);
	vector<pair<double, double>> getRadianAndDistanceFromEnemy(UnitInfo *uInfo, int directCnt);
	vector<int> getEnemiesInAngle(UnitInfo *uInfo, uList enemies, int directCnt, int range);
}