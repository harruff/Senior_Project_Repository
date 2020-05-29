/*
 * Copyright (C) 2019 SAMSUNG SDS <Team.SAIDA@gmail.com>
 *
 * This code is distribued under the terms and conditions from the MIT License (MIT).
 *
 * Authors : Iljoo Yoon, Daehun Jun, Hyunjae Lee, Uk Jo
 */

#pragma once

#include "../Common.h"

namespace MyBot
{
	enum PosChange
	{
		Stop = 0,
		Closer = 1,
		Farther = 2
	};

	class UnitInfo
	{
	public :
		UnitInfo();
		~UnitInfo();
		UnitInfo(Unit);

		void initFrame();
		virtual void Update();

		Unit		unit() const {
			return m_unit;
		}
		UnitType type() const {
			return m_type;
		}
		Position	pos() const {
			return m_lastPosition;
		}
		Position	vPos() const {
			return m_vPosition;
		}
		Position	vPos(int frame) const {
			return m_lastPosition + Position((int)(m_unit->getVelocityX() * frame), (int)(m_unit->getVelocityY() * frame));
		}
		Player		player() const {
			return m_player;
		}
		int			id() const {
			return m_unitID;
		}
		int			hp() const {
			return m_HP + m_Shield;
		}
		int			shield() const {
			return m_Shield;
		}
		int			cooldown() const {
			return m_cooldown;
		}
		bool		isComplete() const {
			return m_completed;
		}
		bool		isMorphing() const {
			return m_morphing;
		}
		bool		isBurrowed() const {
			return m_burrowed;
		}
		void		setFrame(int f = 0) {
			m_LastCommandFrame = TIME + f;
		}
		int			frame() const {
			return m_LastCommandFrame;
		}
		int			expectedDamage() const {
			return m_expectedDamage;
		}
		bool		isBeingRepaired() const {
			return m_beingRepaired;
		}
		bool		isPowered() const {
			return m_isPowered;
		}

		void	recudeEnergy(int energy) {
			m_energy -= min(m_energy, (double)energy);
		}

		int		getEnergy() const {
			return (int)m_energy;
		}

		// Attacker�� this unit�� �����Ҷ� Damage�� ����ؼ� ��������
		void		setDamage(Unit attacker);

		// Building ��
		void		setLift(bool l) {
			m_lifted = l;
		}
		/// true if flying
		bool		getLift() {
			return m_lifted;
		}

		//enemy ��
		const bool isHide() const {
			return m_hide;
		}

		const bool operator == (Unit unit) const
		{
			return m_unitID == unit->getID();
		}
		const bool operator == (const UnitInfo &rhs) const
		{
			return (m_unitID == rhs.m_unitID);
		}
		const bool operator < (const UnitInfo &rhs) const
		{
			return (m_unitID < rhs.m_unitID);
		}

		string getLastSituation() {
			return m_lastSituation;
		}
		void setLastSituation(string situation) {
			m_lastSituation = situation;
		}
		void addLastSituation(string situation) {
			m_lastSituation = m_lastSituation + "," + situation;
		}

		PosChange posChange(UnitInfo *uInfo);

		// Ŭ��ŷ ������ ���Ե��� ����.
		vector<Unit> &getEnemiesTargetMe() {
			return m_enemiesTargetMe;
		}
		Position	getAvgEnemyPos() {
			return m_avgEnemyPosition;
		}
		Unit		getVeryFrontEnemyUnit() {
			return m_veryFrontEnemUnit;
		}
		void		clearAvgEnemyPos() {
			m_avgEnemyPosition = Positions::None;
		}
		void		clearVeryFrontEnemyUnit() {
			m_veryFrontEnemUnit = nullptr;
		}
		int			getLastPositionTime() {
			return m_lastPositionTime;
		}
		Position	getLastSeenPosition() {
			return m_lastSeenPosition;
		}

		bool		isBlocked() const {
			return m_blockedCnt > 25;
		}

		void setLastCommandPosition(Position lastPosition) {
			m_lastCommandPosition = lastPosition;
		}

		Position getLastCommandPosition() {
			return m_lastCommandPosition;
		}

		void setLastAction(int action) {
			m_lastAction = action;
		}

		int getLastAction() {
			return m_lastAction;
		}

		virtual int	getSpaceRemaining() {
			return m_spaceRemaining;
		}

	protected :
		Unit							m_unit;
		UnitType					m_type;
		Player						m_player;
	private :
		int								m_unitID;
		int								m_HP;
		bool							m_beingRepaired;
		int								m_Shield;
		double						m_energy;
		int								m_LastCommandFrame;

		int								m_expectedDamage; // ���� ���� Damage�� �����ϱ� ����.

		int								m_lastPositionTime; // m_lastPosition �� ó�� ������ �ð�
		Position						m_lastSeenPosition; // m_lastPosition �� �����ϳ� Unknown ���� �ٲ��� ������ ����.
		Position						m_lastPosition;
		Position						m_vPosition;
		bool							m_completed;
		bool							m_morphing;
		bool							m_burrowed;
		bool							m_canBurrow;
		bool							m_hide;
		int								m_spaceRemaining;
		bool							m_isPowered;
		queue<bool>						m_blockedQueue;
		int								m_blockedCnt; // 120 frame �� �渷�� ���� Ƚ��
		int								m_cooldown;
		// burrow ���� unknown position ���� �����Ű�� ���� ���.
		int								m_nearUnitFrameCnt;
		int								m_lastNearUnitFrame;

		// �ǹ��� lift Ȯ��
		bool							m_lifted;

		vector<Unit>					m_enemiesTargetMe;
		Position						m_avgEnemyPosition;
		Unit							m_veryFrontEnemUnit;

		// BWML render ������ ���ؼ� �����.
		Position						m_lastCommandPosition;
		int								m_lastAction;

		// for debug
		string m_lastSituation = "";
	};
}